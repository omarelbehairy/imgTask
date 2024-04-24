import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui
from canny_edge_detector_ui import Ui_Dialog
from PIL import Image


class CannyEdgeDetectorApp(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Connect button signals to slots
        self.Loadbtn.clicked.connect(self.load_image)
        self.pushButton_2.clicked.connect(self.detect_edges)

        # Connect slider signals to slots
        self.LowerSlider.valueChanged.connect(self.update_lower_threshold)
        self.UpperSlider.valueChanged.connect(self.update_upper_threshold)

        # Initialize image variables
        self.original_image = None
        self.edge_image = None
        self.lower_threshold = 50  # Initial lower threshold
        self.upper_threshold = 80  # Initial upper threshold

    def update_lower_threshold(self, value):
        self.lower_threshold = value

    def update_upper_threshold(self, value):
        self.upper_threshold = value

    def load_image(self):
        # Open a file dialog to select an image file
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        file_dialog.setViewMode(QtWidgets.QFileDialog.List)
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)

        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            # Read the selected image file
            self.original_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            # Display the original image
            self.display_image(self.original_image, self.originalImage)

    def detect_edges(self):
        if self.original_image is not None:
            # Perform Canny edge detection with updated thresholds
            detected_edges = self.canny_edge_detection(
                self.original_image, 9, self.lower_threshold, self.upper_threshold)
            cv2.imshow("Detected Edges", detected_edges)
            # Display the detected edges
            self.display_image(detected_edges, self.edgeImage)

    def grayscale(self, image):
        """
        Converts the input image to grayscale.

        Parameters:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Grayscale image.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def gaussian_blur(self, image, kernel_size):
        """
        Applies Gaussian blur to the input image.

        Parameters:
            image (numpy.ndarray): Input image.
            kernel_size (int): Size of the Gaussian kernel.

        Returns:
            numpy.ndarray: Blurred image.
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def sobel_gradients(self, image):
        """
        Calculates the gradients in x and y directions using Sobel operators.

        Parameters:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Gradient in x direction.
            numpy.ndarray: Gradient in y direction.
        """
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        return gradient_x, gradient_y

    def gradient_magnitude_direction(self, gradient_x, gradient_y):
        """
        Calculates gradient magnitude and direction.

        Parameters:
            gradient_x (numpy.ndarray): Gradient in x direction.
            gradient_y (numpy.ndarray): Gradient in y direction.

        Returns:
            numpy.ndarray: Gradient magnitude.
            numpy.ndarray: Gradient direction.
        """
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
        print("-----------------------------------")
        print(gradient_direction)
        print("-----------------")
        return gradient_magnitude, gradient_direction

    def non_maximum_suppression(self, gradient_magnitude, gradient_direction):
        """
        Applies non-maximum suppression to thin the edges.

        Parameters:
            gradient_magnitude (numpy.ndarray): Gradient magnitude.
            gradient_direction (numpy.ndarray): Gradient direction.

        Returns:
            numpy.ndarray: Suppressed image.
        """
        suppressed = np.zeros_like(gradient_magnitude)
        rows, cols = gradient_magnitude.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                angle = gradient_direction[i, j]
                # Determine the neighbors based on the gradient direction
                if (0 < angle <= 45) or (180 < angle <= 225) or (-180 < angle <= -135):
                    neighbors = [gradient_magnitude[i-1, j - 1],
                                 gradient_magnitude[i+1, j + 1]]
                elif (45 < angle <= 90) or (225 < angle <= 270) or (-135 < angle <= -90):
                    neighbors = [gradient_magnitude[i, j - 1],
                                 gradient_magnitude[i, j + 1]]
                elif (90 < angle <= 135) or (270 < angle <= 315) or (-90 < angle <= -45):
                    neighbors = [gradient_magnitude[i - 1, j+1],
                                 gradient_magnitude[i + 1, j-1]]
                elif (135 < angle <= 180) or (315 < angle <= 360) or (-45 < angle <= 0):
                    neighbors = [gradient_magnitude[i + 1, j],
                                 gradient_magnitude[i - 1, j]]
                # Suppress non-maximum pixels
                if gradient_magnitude[i, j] >= max(neighbors):
                    suppressed[i, j] = gradient_magnitude[i, j]
        return suppressed

    def double_thresholding(self, suppressed, low_threshold, high_threshold):
        """
        Applies double thresholding to identify strong, weak, and non-edge pixels.

        Parameters:
            suppressed (numpy.ndarray): Suppressed image.
            low_threshold (float): Low threshold value.
            high_threshold (float): High threshold value.

        Returns:
            numpy.ndarray: Image containing strong edges.
            numpy.ndarray: Image containing weak edges.
        """
        strong_edges = suppressed > high_threshold
        weak_edges = (suppressed >= low_threshold) & (
            suppressed <= high_threshold)
        return strong_edges, weak_edges

    def edge_tracking(self, edges, weak_edges):
        """
        Performs edge tracking by hysteresis to connect weak edges to strong edges.

        Parameters:
            edges (numpy.ndarray): Image containing strong edges.
            weak_edges (numpy.ndarray): Image containing weak edges.

        Returns:
            numpy.ndarray: Final image with connected edges.
        """
        rows, cols = edges.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if weak_edges[i, j]:
                    # Check if any neighboring pixel is a strong edge
                    if np.any(edges[i - 1:i + 2, j - 1:j + 2]):
                        edges[i, j] = 255
        return edges

    def hysteresis_thresholding(self, image, low_threshold, high_threshold):
        """
        Applies hysteresis thresholding for edge tracking.

        Args:
            image: The image with suppressed non-maximum gradients.
            low_threshold: The lower threshold for edge tracking.
            high_threshold: The upper threshold for edge tracking.

        Returns:
            The edge image as a NumPy array (binary image with detected edges).
        """
        rows, cols = image.shape
        edges = np.zeros_like(image)

        # Mark strong edges based on high threshold
        strong_edges = (image >= high_threshold)

        # Mark weak edges based on low threshold
        weak_edges = (image >= low_threshold) & (image < high_threshold)

        # Define 8-connected neighborhood
        neighbors = [(i, j) for i in range(-1, 2)
                     for j in range(-1, 2) if (i != 0 or j != 0)]

        # Function to recursively track weak edges connected to strong edges
        def track_weak_edges(i, j):
            if edges[i, j] == 1:
                return
            edges[i, j] = 1
            for dx, dy in neighbors:
                x, y = i + dx, j + dy
                if 0 <= x < rows and 0 <= y < cols and weak_edges[x, y]:
                    track_weak_edges(x, y)

        # Track weak edges connected to strong edges
        for i in range(rows):
            for j in range(cols):
                if strong_edges[i, j]:
                    track_weak_edges(i, j)

        return edges

    def canny_edge_detection(self, image, kernel_size, low_threshold, high_threshold):
        """
        Performs Canny edge detection on the input image.

        Parameters:
            image (numpy.ndarray): Input image.
            kernel_size (int): Size of the Gaussian kernel for blurring.
            low_threshold (float): Low threshold for double thresholding.
            high_threshold (float): High threshold for double thresholding.

        Returns:
            numpy.ndarray: Image containing the detected edges.
        """
        # Step 1: Grayscale conversion
        gray_image = self.grayscale(image)
        # Step 2: Gaussian blur
        blurred = self.gaussian_blur(gray_image, kernel_size)
        # Step 3: Gradient calculation
        gradient_x, gradient_y = self.sobel_gradients(blurred)
        # Step 4: Gradient magnitude and direction
        magnitude, direction = self.gradient_magnitude_direction(
            gradient_x, gradient_y)
        # Step 5: Non-maximum suppression
        suppressed = self.non_maximum_suppression(magnitude, direction)
        # Step 6: Double thresholding
        strong_edges, weak_edges = self.double_thresholding(
            suppressed, low_threshold, high_threshold)
        # Step 7: Edge tracking by hysteresis
        edges = self.hysteresis_thresholding(
            suppressed, low_threshold, high_threshold)
        return edges

    def display_image(self, image, label_widget):
        # Convert the OpenCV image to QImage
        if label_widget == self.edgeImage:  # Grayscale image
            q_img = QtGui.QImage(
                image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_Grayscale8)
        else:  # Color image
            if image.shape[2] == 3:
                q_img = QtGui.QImage(
                    image.data, image.shape[1], image.shape[0], image.shape[1] * 3, QtGui.QImage.Format_RGB888).rgbSwapped()
            elif image.shape[2] == 4:
                q_img = QtGui.QImage(
                    image.data, image.shape[1], image.shape[0], image.shape[1] * 4, QtGui.QImage.Format_ARGB32)
            else:
                raise ValueError(
                    "Unsupported image format: {} channels".format(image.shape[2]))

        # Display the QImage in the QLabel widget
        pixmap = QtGui.QPixmap.fromImage(q_img)
        label_widget.setPixmap(pixmap)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = CannyEdgeDetectorApp()
    mainWindow.show()
    sys.exit(app.exec_())
