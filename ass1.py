# Assessment 1: Applying three lossless compression techniques—Run-Length Encoding (RLE), Golomb Coding, and LZW—on a sample image

from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

# ----------------------------
# Load and show original image
# ----------------------------
image_path = "6307733789072887313.png"   # make sure image is in same folder

original = Image.open(image_path).convert("L")

# Show original image
plt.imshow(original, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.show()

# ----------------------------
# Resize image
# ----------------------------
img = original.resize((64, 64))

# Show resized image
plt.imshow(img, cmap='gray')
plt.title("Resized Image (64x64)")
plt.axis('off')
plt.show()

# ----------------------------
# Convert to array
# ----------------------------
pixels = np.array(img)
data = pixels.flatten().tolist()

print("Total pixels:", len(data))


# ----------------------------
# 1. Run-Length Encoding (RLE)
# ----------------------------
def rle_encode(data):
    encoding = []
    prev = data[0]
    count = 1

    for i in range(1, len(data)):
        if data[i] == prev:
            count += 1
        else:
            encoding.append((prev, count))
            prev = data[i]
            count = 1

    encoding.append((prev, count))
    return encoding


rle_output = rle_encode(data)
print("RLE size:", len(rle_output))


# ----------------------------
# 2. Golomb Coding
# ----------------------------
def golomb_encode(numbers, m):
    result = ""
    b = math.ceil(math.log2(m))

    for n in numbers:
        q = n // m
        r = n % m

        unary = '1' * q + '0'
        binary = format(r, f'0{b}b')

        result += unary + binary + " "

    return result.strip()


counts = [count for _, count in rle_output]
golomb_output = golomb_encode(counts, m=4)

print("Golomb encoded length:", len(golomb_output))


# ----------------------------
# 3. LZW Encoding
# ----------------------------
def lzw_encode(data):
    dictionary = {str(i): i for i in set(data)}
    dict_size = len(dictionary)

    w = ""
    result = []

    for value in data:
        k = str(value)
        wk = w + "," + k if w else k

        if wk in dictionary:
            w = wk
        else:
            result.append(dictionary[w])
            dictionary[wk] = dict_size
            dict_size += 1
            w = k

    if w:
        result.append(dictionary[w])

    return result


lzw_output = lzw_encode(data)
print("LZW size:", len(lzw_output))


# ----------------------------
# Compression Summary
# ----------------------------
original_size = len(data)

print("\n--- Compression Summary ---")
print("Original size:", original_size)
print("RLE size:", len(rle_output))
print("LZW size:", len(lzw_output))

