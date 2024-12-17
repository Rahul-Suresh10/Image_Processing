import cv2 as cv
import numpy as np

# Load the image
image = cv.imread('./photos/positive_sample.jpeg')
if image is None:
    print("Error: Could not read the image file. Check the file path.")
    exit()

# Resize the image and crop the region of interest
resized_img = cv.resize(image, (500, 500), interpolation=cv.INTER_AREA)
cropped_img = resized_img[210:400, 195:317]

# Convert the cropped image to HSV color space and split channels
hsv = cv.cvtColor(cropped_img, cv.COLOR_BGR2HSV)
h, s, v = cv.split(hsv)

# Enhance Saturation and Value channels for better visibility of faint colors
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
s = clahe.apply(s)
v = clahe.apply(v)

# Increase saturation for pixels above a threshold
s = np.where(s > 20, np.minimum(s + 30, 255), s)

# Merge enhanced HSV channels and convert back to BGR for improved color visibility
enhanced_hsv = cv.merge((h, s, v))
enhanced_bgr = cv.cvtColor(enhanced_hsv, cv.COLOR_HSV2BGR)
cv.imshow("Enhanced",enhanced_bgr)
# Step 1: Detect the C line (no changes needed here as it is already working)
# Define HSV color range for red shades for C line
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 50, 50])
upper_red2 = np.array([180, 255, 255])

# Create masks for red color ranges for C line detection
mask_c_line1 = cv.inRange(enhanced_hsv, lower_red1, upper_red1)
mask_c_line2 = cv.inRange(enhanced_hsv, lower_red2, upper_red2)
mask_c_line = mask_c_line1 + mask_c_line2

# Apply morphological operations to refine the mask
kernel = np.ones((3, 3), np.uint8)
mask_c_line = cv.morphologyEx(mask_c_line, cv.MORPH_CLOSE, kernel)
mask_c_line = cv.morphologyEx(mask_c_line, cv.MORPH_OPEN, kernel)

# Find contours in the C line mask
contours_c, _ = cv.findContours(mask_c_line, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
c_line_detected = False
c_line_rect = None

# Loop through contours to find the C line on the left side
for contour in contours_c:
    if cv.contourArea(contour) > 50:  # Area threshold for the C line
        x, y, w, h = cv.boundingRect(contour)
        if x + w < cropped_img.shape[1] // 2:  # Ensure it's on the left side (C line)
            cv.rectangle(cropped_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box for C line
            c_line_detected = True
            c_line_rect = (x, y, w, h)
            print("C line detected. Location:", (x, y, w, h))
            break

# Display the result for C line detection
cv.imshow('Cropped Image with Detected C Line', cropped_img)
cv.imshow('C Line Mask', mask_c_line)

if not c_line_detected:
    print("C line not detected; skipping T line detection.")
else:
    # Step 2: Detect the faint T line by isolating faint red regions in the enhanced image
    # Convert the enhanced image to HSV color space
    hsv_enhanced = cv.cvtColor(enhanced_bgr, cv.COLOR_BGR2HSV)

    # Define HSV ranges for faint red colors (corresponding to RGB (232, 190, 160))
    # These values should capture faint red tones for the T line
    lower_faint_red = np.array([10, 50, 100])  # These values should capture faint red tones
    upper_faint_red = np.array([20, 255, 255])

    # Create a mask for the faint red color range (faint T line)
    mask_faint_red = cv.inRange(hsv_enhanced, lower_faint_red, upper_faint_red)

    # Optional: Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask_faint_red = cv.morphologyEx(mask_faint_red, cv.MORPH_CLOSE, kernel)
    mask_faint_red = cv.morphologyEx(mask_faint_red, cv.MORPH_OPEN, kernel)

    # Use the mask to extract the faint red regions (T line) from the enhanced BGR image
    faint_red_isolated = cv.bitwise_and(enhanced_bgr, enhanced_bgr, mask=mask_faint_red)

    # Display the results for T line detection
    cv.imshow('Faint Red Pixels Isolated (T Line)', faint_red_isolated)
    cv.imshow('Faint Red Mask', mask_faint_red)

    # Step 3: Estimate the Intensity of the T line in 5 positions (about 0.25 x 0.25 mm areas)
    # Assume the positions for T line intensity calculation are centered around the T line area

    t_line_intensities = []
    for i in range(5):
        # Simulate sampling intensity at 5 positions (manual positions based on the detected region)
        x_offset = 10 + i * 5  # Adjust x_offset to simulate different sample points
        y_offset = 20 + i * 5  # Adjust y_offset to simulate different sample points
        region = cropped_img[y_offset:y_offset+10, x_offset:x_offset+10]  # Sampling a 10x10 area
        mean_intensity = np.mean(region)  # Mean intensity of this region
        t_line_intensities.append(mean_intensity)
        print(f"T line intensity at position {i+1}: {mean_intensity}")

    t_line_mean_intensity = np.mean(t_line_intensities)
    print("Mean T line intensity: ", t_line_mean_intensity)

    # Step 4: Estimate the Background intensity from 10 positions
    background_intensities = []
    for i in range(10):
        # Simulate sampling intensity at 10 random background positions
        x_offset = 5 + i * 5
        y_offset = 10 + i * 5
        region = cropped_img[y_offset:y_offset+10, x_offset:x_offset+10]  # Sampling a 10x10 area
        mean_intensity = np.mean(region)  # Mean intensity of this region
        background_intensities.append(mean_intensity)
        print(f"Background intensity at position {i+1}: {mean_intensity}")

    background_mean_intensity = np.mean(background_intensities)
    print("Mean background intensity: ", background_mean_intensity)

    # Step 5: Subtract Background intensity from T line intensity
    t_line_adjusted_intensity = t_line_mean_intensity - background_mean_intensity
    print("Adjusted T line intensity (after background subtraction): ", t_line_adjusted_intensity)

    # Step 6: Fit the T line intensity to a linear curve equation and calculate the analyte value
    # For this example, let's assume a linear relation between intensity and analyte concentration
    # Example linear curve: analyte_value = slope * adjusted_intensity + intercept

    slope = 0.5  # Example slope
    intercept = 2.0  # Example intercept

    analyte_value = slope * t_line_adjusted_intensity + intercept
    print("Calculated analyte value: ", analyte_value)

# Wait for key press to close windows
cv.waitKey(0)
cv.destroyAllWindows()
