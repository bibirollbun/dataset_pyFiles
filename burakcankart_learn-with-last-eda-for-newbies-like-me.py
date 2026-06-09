# As all of you already know, we are starting to import these libraries using basic syntax.

import numpy as np
import pandas as pd
import h5py


# Then, we copy our file's path and paste it to upload.

h5file = h5py.File("/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5", "r")


# This is a different style of printing. Let's look at what our file has:

with h5py.File("/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5", "r") as file:
    for name in file:
        print(name, "/")
        for j in file[name]:
            print("---",j, "//")
            for k in file[name][j]:
                print("------",k)


# We can do the same thing as above, with a function. Let's see:

def get_around(group, level=0):
    for name in group:
        thing = group[name]
        print("--" * level + name)
        if isinstance(thing, h5py.Group):
            get_around(thing, level + 1)


with h5py.File("/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5", "r") as file:
    get_around(file)


with h5py.File("/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5", "r") as file:
    def load_data(file, group_path):
        return {s_name: file[group_path][s_name][:] for s_name in file[group_path]}
    train_images = load_data(file, "images/Train")
    test_images = load_data(file, "images/Test")
    train_spots = load_data(file, "spots/Train")
    test_spots = load_data(file, "spots/Test")


# What are these file types? What do they look like?

print(type(train_images))
print(type(train_images["S_1"]))


train_images["S_1"][0]


print(type(train_images["S_1"].item(0)))
train_images["S_1"].item(0)


import matplotlib.pyplot as plt


# How many dimensions does it have?
train_images["S_1"].ndim


ex_arr = np.array([[1, 2], [3, 4]])
print(sum(ex_arr))  # result: the sum of that array is â†’ 10


print(ex_arr.sum())           # result: 10
print(ex_arr.sum(axis=0))    # result: total by column â†’ [4, 6]
print(ex_arr.sum(axis=1))     # result: total by row â†’ [3, 7]


# The "plot" function of NumPy doesn't work with 2D data.

key = "S_1"
img = train_images[key]
plt.imshow(img)


flattened = train_images["S_1"].reshape(-1)
plt.plot(flattened)


img.shape

# 2000 height, 1974 width, 3 RGB colors.
# So it has (2000 x 1974 x 3) value. -> 11.844.000


print(train_images["S_1"][0].shape)


# This is the first pixel of the "height" and it has a 1974 width value with the 3 different colors: 
print(train_images["S_1"][0])

print("--------------------")

# If you want to focus on RGB colors for the first pixel of the image, you could use:
print(train_images["S_1"][0][0])

print("--------------------")

# If you want to focus on the RED color of the first pixel of the image, you could use:
print(train_images["S_1"][0][0][0])

print("--------------------")


# I just tried what would happen if I ran this code, nothing important here:

plt.plot(img[444][1])


plt.imshow(img[:,:,1])

# If I choose a color to see the first image:


def visualizing(img):
    fig = plt.figure(figsize=(10,7))

    plt.subplot(3, 3, 1)  # 2 rows, 2 columns, first position
    plt.imshow(img)  
    plt.axis('off')  # Hide the axis labels
    plt.title("Original") 
    
    # Add the second image to the figure (top-right position)
    plt.subplot(3, 3, 2)  # 2 rows, 2 columns, second position
    plt.imshow(img[:,:,0],cmap="Greens")  
    plt.axis('off')  # Hide the axis labels
    plt.title("Green") 
    
    # Add the third image to the figure (bottom-left position)
    plt.subplot(3, 3, 3)  # 2 rows, 2 columns, third position
    plt.imshow(img[:,:,0],cmap="Reds") 
    plt.axis('off')  # Hide the axis labels
    plt.title("Red")  
    # Add the fourth image to the figure (bottom-right position)
    plt.subplot(3, 3, 4)  # 2 rows, 2 columns, fourth position
    plt.imshow(img[:,:,0],cmap="Blues")  
    plt.axis('off')  # Hide the axis labels
    plt.title("Blue")  
    
    plt.subplot(3,3,5)
    plt.imshow(img[:,:,0], cmap="Purples_r")
    plt.axis("off")
    plt.title("Purple")

    plt.subplot(3,3,6)
    plt.imshow(img[:,:,0], cmap="turbo")
    plt.title("Turbo")
    
    # Plotlib is fun... Really.ğŸ¤©


visualizing(img)


from matplotlib import colormaps
print(colormaps)
# To see the colors you can use on plotlib


spot = train_spots[key]


spot.shape


spot.dtype


img.dtype


# How many X, Y, and "C" values are there in a spot variable?

print(spot["x"].sum())
print(spot["y"].sum())
print(spot["C1"].sum())
print(spot["C2"].sum())
print("----------- \n")
print(spot["C35"].sum())


spot["C1"]


fig = plt.figure(figsize=(10,7))

plt.imshow(img)  
plt.axis('off')  # Hide the axis labels
plt.title("Original") 
plt.scatter(spot["x"],spot["y"])

# When we put these dots on the image.. Image looks like this.. I GOT IT.


plt.imshow(img)  
plt.axis('off')  # Hide the axis labels
plt.title("Original") 
plt.scatter(spot["x"],spot["y"], s=10, c="red")


def imaginary(img, number, figsize, s):
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.scatter(spot["x"], spot["y"], c=spot[f"C{number}"], cmap="coolwarm", s=s)
    plt.colorbar()


imaginary(img, 35, (20,15), 5)


# Threshold filter... Print where "C35" is higher than "0.5" value... 
mask = spot["C35"] > 0.5


plt.imshow(img)
plt.scatter(spot["x"][mask], spot["y"][mask], c="red", s=1)


# So, where are these points? What are their values?
img[spot["y"][mask], spot["x"][mask]]


def plot_it(img, number, figsize_1, figsize_2, s):
    fig = plt.figure(figsize=(figsize_1, figsize_2))

    plt.suptitle(f"SLIDE: {key} SPOT: C{number}")
    plt.subplot(1, 3, 1)  
    plt.imshow(img)  
    plt.axis('off')  
    plt.title("Original")
    plt.colorbar()

    plt.subplot(1, 3, 2) 
    plt.imshow(img)  
    plt.axis('off')  
    plt.title("Colorized") 
    plt.scatter(spot["x"],spot["y"], s=s, c="red")
    plt.colorbar()

    plt.subplot(1, 3, 3)  
    plt.imshow(img)  
    plt.axis('off')  
    plt.title("Detailed") 
    plt.scatter(spot["x"], spot["y"], c=spot[f"C{number}"], cmap="coolwarm", s=s)
    plt.colorbar()


key = "S_1"
img = train_images[key]
spot = train_spots[key]

# Values can be changed. I choose them randomly.
plot_it(img, 24, 20, 5, 1)


key = "S_2"
img = train_images[key]
spot = train_spots[key]

plot_it(img, 35, 20, 5, 1)


key = "S_3"
img = train_images[key]
spot = train_spots[key]

plot_it(img, 19, 20, 5, 1)


key = "S_4"
img = train_images[key]
spot = train_spots[key]

plot_it(img, 7, 20, 5, 1)


key = "S_5"
img = train_images[key]
spot = train_spots[key]

plot_it(img, 1, 20, 5, 1)


key = "S_6"
img = train_images[key]
spot = train_spots[key]

plot_it(img, 4, 20, 5, 1)


spot_test = test_spots["S_7"]
image_test = test_images["S_7"]


spot_test.dtype


image_test.shape


fig = plt.figure(figsize=(20,5))

plt.suptitle("Test Data")
plt.subplot(1,3,1)
plt.imshow(image_test)
plt.axis("off")
plt.title("Orjinal")

plt.subplot(1,3,2)
plt.imshow(image_test)
plt.scatter(spot_test["x"], spot_test["y"], c="red", s=1)
plt.axis("off")
plt.title("Colorized")
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(image_test)
plt.scatter(spot_test["x"], spot_test["y"], c=spot_test["Test_Set"], cmap="coolwarm", s=1)
plt.axis("off")
plt.title("Detailed")
plt.colorbar()


# C1 to C35..

x = train_spots["S_1"]["x"]
Y = train_spots["S_1"]["y"]

temp_spot = train_spots["S_1"]

c_values = np.stack([temp_spot[name] for name in temp_spot.dtype.names if name.startswith("C")], axis=1)
c_sum = c_values.sum(axis=1)


c_sum


c_sum.sum()


print(temp_spot[0], "\n")
print(sum(temp_spot[0]), "\n")
print(sum(temp_spot[0]) - (temp_spot[0]["x"] + temp_spot[0]["y"]))


key = "S_1"
img = train_images[key]
spot = train_spots[key]

plt.imshow(img)
plt.scatter(x, Y, c=c_sum, cmap="coolwarm", s=1)
plt.colorbar()


plt.scatter(x, Y, c=c_sum, cmap="coolwarm", s=1)
plt.colorbar()


plt.hist(c_sum)
# This is nonsense. Whatever...

