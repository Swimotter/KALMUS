# Third-party modules
import cv2
import matplotlib.pyplot as plt
import pytest


# Test data shared between module tests
@pytest.fixture(scope="session")
def get_test_color_image():
    return plt.imread("tests/test_data/test_color_image.jpg", format="jpeg")


@pytest.fixture(scope="session")
def get_test_gray_image():
    clr_image = plt.imread("tests/test_data/test_color_image.jpg", format="jpeg")
    gray_image = cv2.cvtColor(clr_image, cv2.COLOR_RGB2GRAY)
    return gray_image
