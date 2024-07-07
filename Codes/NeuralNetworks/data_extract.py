# import os
# import cv2

# images = []
# with open("data.csv","a") as f:
#     for i in range(32*32):
#         f.write(str(i))
#         f.write(",")
#     f.write("\n")           

# def find(name, path):
#     with open("data.csv","a") as f:
#         for root, dirs, files in os.walk(path):
#             for i in files:
#                 if name in i:
#                     path = os.path.join(root,i)
#                     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#                     img = cv2.resize(img,(32,32))
#                     # print(img)
#                     for i in range(len(img)):
#                         for j in range(len(img[0])):
#                             temp = int(img[i][j])
#                             f.write(str(temp))
#                             f.write(",")
#                     f.write("\n")

# find(".png",".")
# # images = cv2.imread(r"C:\Users\Haarit\Desktop\MachineLearning\Codes\NeuralNetworks\DevanagariHandwrittenCharacterDataset\Test\character_1_ka\1339.png")
# print(images)




import os
import cv2

# Initialize an empty list to store images
images = []

# Open (or create) a CSV file named "data.csv" in append mode
with open("data.csv", "w") as f:
    # Write the header for a 32x32 image (1024 values separated by commas)
    for i in range(32 * 32 + 1):
        f.write(str(i))
        f.write(",")
    f.write("\n")  # End of the header line

datafory = {
    "character_1_ka":0,
    "character_2_kha":1,
    "character_3_ga":2,
    "character_4_gha":3,
    "character_5_kna":4,
    "character_6_cha":5,
    "character_7_chha":6,
    "character_8_ja":7,
    "character_9_jha":8,
    "character_10_yna":9,
    "character_11_taamatar":10,
    "character_12_thaa":11,
    "character_13_daa":12,
    "character_14_dhaa":13,
    "character_15_adna":14,
    "character_16_tabala":15,
    "character_17_tha":16,
    "character_18_da":17,
    "character_19_dha":18,
    "character_20_na":19,
    "character_21_pa":20,
    "character_22_pha":21,
    "character_23_ba":22,
    "character_24_bha":23,
    "character_25_ma":24,
    "character_26_yaw":25,
    "character_27_ra":26,
    "character_28_la":27,
    "character_29_waw":28,
    "character_30_motosaw":29,
    "character_31_petchiryakha":30,
    "character_32_patalosaw":31,
    "character_33_ha":32,
    "character_34_chhya":33,
    "character_35_tra":34,
    "character_36_gya":35,
    "digit_0":36,
    "digit_1":37,
    "digit_2":38,
    "digit_3":39,
    "digit_4":40,
    "digit_5":41,
    "digit_6":42,
    "digit_7":43,
    "digit_8":44,
    "digit_9":45
}

# Function to find and process images with a given name pattern in a specified path
def find(name, path):
    # Open (or create) the CSV file in append mode
    with open("data.csv", "a") as f:
        # Traverse the directory tree starting at 'path'
        for root, dirs, files in os.walk(path):
            # Iterate over the list of files
            for i in files:
                # Check if the file name contains the specified pattern
                if name in i:
                    # Get the full path of the image file
                    path = os.path.join(root, i)

                    for key,value in datafory.items():
                        if(key in path):
                            f.write(str(value))
                            f.write(",")

                    # Read the image in grayscale mode
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    # Resize the image to 32x32 pixels
                    img = cv2.resize(img, (32, 32))
                    # Optional: Print the image array for debugging
                    # print(img)
                    # Iterate over the rows of the image
                    for i in range(len(img)):
                        # Iterate over the columns of the image
                        for j in range(len(img[0])):
                            # Convert the pixel value to an integer
                            temp = int(img[i][j])
                            # Write the pixel value to the CSV file
                            f.write(str(temp))
                            f.write(",")
                    f.write("\n")  # End of the image data line

# Call the function to find and process images with ".png" in the current directory
find(".png", ".")

# Print the list of images (currently empty since we don't append images to this list)
print(images)
