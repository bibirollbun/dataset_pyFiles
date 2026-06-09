! pip install zarr


import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zarr


# Load the first zarr.
z_ts_6_4 = zarr.open('/kaggle/input/czii-cryo-et-object-identification/test/static/ExperimentRuns/TS_6_4/VoxelSpacing10.000/denoised.zarr', mode='r')
z_ts_6_4_iso = zarr.open('/kaggle/input/czii-cryo-et-object-identification/train/static/ExperimentRuns/TS_6_4/VoxelSpacing10.000/isonetcorrected.zarr', mode='r')
z_ts_6_4_dcon = zarr.open('/kaggle/input/czii-cryo-et-object-identification/train/static/ExperimentRuns/TS_6_4/VoxelSpacing10.000/ctfdeconvolved.zarr', mode='r')
z_ts_6_4_wbp = zarr.open('/kaggle/input/czii-cryo-et-object-identification/train/static/ExperimentRuns/TS_6_4/VoxelSpacing10.000/wbp.zarr', mode='r')


print(f"{z_ts_6_4} :: 0:{z_ts_6_4[0].shape} ,1: {z_ts_6_4[1].shape} ,2:{z_ts_6_4[2].shape}")
print(f"{z_ts_6_4_iso} :: 0:{z_ts_6_4_iso[0].shape} ,1: {z_ts_6_4_iso[1].shape} ,2:{z_ts_6_4_iso[2].shape}")
print(f"{z_ts_6_4_dcon} :: 0:{z_ts_6_4_dcon[0].shape} ,1: {z_ts_6_4_dcon[1].shape} ,2:{z_ts_6_4_dcon[2].shape}")
print(f"{z_ts_6_4_wbp} :: 0:{z_ts_6_4_wbp[0].shape} ,1: {z_ts_6_4_wbp[1].shape} ,2:{z_ts_6_4_wbp[2].shape}")


fig = plt.figure(figsize=(6.3,6.3))
_=plt.imshow(z_ts_6_4[0][64])


fig = plt.figure(figsize=(30,60))
for i in range(184):
    ax = plt.subplot(20,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(z_ts_6_4[0][i])


fig = plt.figure(figsize=(6.3,6.3))
_=plt.imshow(z_ts_6_4[1][0])


fig = plt.figure(figsize=(30,60))
for i in range(92):
    ax = plt.subplot(20,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(z_ts_6_4[1][i])


fig = plt.figure(figsize=(6.3,6.3))
_=plt.imshow(z_ts_6_4[2][0])


fig = plt.figure(figsize=(30,60))
for i in range(46):
    ax = plt.subplot(20,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(z_ts_6_4[2][i])


title_list = ['Denoised','IsoNet Corrected','CTF Deconvolved','Weight Back Projection']
img_list = [z_ts_6_4,z_ts_6_4_iso,z_ts_6_4_dcon,z_ts_6_4_wbp]

fig = plt.figure(figsize=(10,10))
for i,title in enumerate(title_list):
    ax = plt.subplot(2,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.imshow(img_list[i][0][62])


fig = plt.figure(figsize=(10,10))
for i,name in enumerate(title_list):
    ax = plt.subplot(2,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.title(name)
    plt.imshow(img_list[i][0][62],cmap='gray')


import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 15, 7, 20, 12]

plt.scatter(x, y, color='blue', marker='o')  # Scatter plot with blue circles
plt.xlabel("X-axis Label")
plt.ylabel("Y-axis Label")
plt.title("Scatter Plot Example")
plt.show()



#file = open('/kaggle/input/czii-cryo-et-object-identification/train/overlay/ExperimentRuns/TS_6_4/Picks/ribosome.json')
#file.read()
'''
'{
"pickable_object_name": "ribosome", 
"user_id": "curation", 
"session_id": "0", 
"run_name": "TS_6_4", 
"voxel_spacing": null, 
"unit": "angstrom", 
"points": [
    {"location": {"x": 5274.903, "y": 5288.121, "z": 619.798}, 
     "transformation_": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 
     "instance_id": 0}, 
    {"location": {"x": 5493.057, "y": 5181.127, "z": 726.624}, 
     "transformation_": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 
     "instance_id": 0}, 
    {"location": {"x": 5656.951, "y": 5168.655, "z": 421.572}, 
     "transformation_": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 
     "instance_id": 0}, 
    
'''


ribosome_x = []
ribosome_y = []

file = open('/kaggle/input/czii-cryo-et-object-identification/train/overlay/ExperimentRuns/TS_6_4/Picks/ribosome.json')
for p in json.loads(file.read())['points']:
    z=float(p["location"]["z"])
    if z >= 600 and z < 650 :
        ribosome_x.append(float(p["location"]["x"])/10)
        ribosome_y.append(float(p["location"]["y"])/10)
        print(p["location"])
    


fig = plt.figure(figsize=(10,8))
for i in range(2):
    ax = plt.subplot(1,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(z_ts_6_4[0][62], cmap='gray', vmin=-0.00005, vmax=0.00005)
    if i%2==1:
        plt.scatter(ribosome_x,ribosome_y, edgecolor='red',facecolor='none')
    


virus_x = []
virus_y = []

file = open('/kaggle/input/czii-cryo-et-object-identification/train/overlay/ExperimentRuns/TS_6_4/Picks/virus-like-particle.json')
for p in json.loads(file.read())['points']:
    z=float(p["location"]["z"])
    if z >= 670 and z < 700 :
        virus_x.append(float(p["location"]["x"])/10)
        virus_y.append(float(p["location"]["y"])/10)
        print(p["location"])


fig = plt.figure(figsize=(10,8))
for i in range(2):
    ax = plt.subplot(1,2,i+1)
    #plt.xticks([])
    #plt.yticks([])
    plt.imshow(z_ts_6_4[0][68], cmap='gray', vmin=-0.00005, vmax=0.00005)
    if i%2==1:
        plt.scatter(virus_x,virus_y, edgecolor='red',facecolor='none')
    


apo_ferritin_x = []
apo_ferritin_y = []

file = open('/kaggle/input/czii-cryo-et-object-identification/train/overlay/ExperimentRuns/TS_6_4/Picks/apo-ferritin.json')
for p in json.loads(file.read())['points']:
    z=float(p["location"]["z"])
    if z >= 400 and z < 450 :
        apo_ferritin_x.append(float(p["location"]["x"])/10)
        apo_ferritin_y.append(float(p["location"]["y"])/10)
        print(p["location"])


fig = plt.figure(figsize=(10,8))
for i in range(2):
    ax = plt.subplot(1,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(z_ts_6_4[0][43], cmap='gray', vmin=-0.00005, vmax=0.00005)
    if i%2==1:
        plt.scatter(apo_ferritin_x, apo_ferritin_y, edgecolor='red',facecolor='none')
    


beta_galactosidase_x = []
beta_galactosidase_y = []

file = open('/kaggle/input/czii-cryo-et-object-identification/train/overlay/ExperimentRuns/TS_6_4/Picks/beta-galactosidase.json')
for p in json.loads(file.read())['points']:
    z=float(p["location"]["z"])
    if z >= 450 and z < 500 :
        beta_galactosidase_x.append(float(p["location"]["x"])/10)
        beta_galactosidase_y.append(float(p["location"]["y"])/10)
        print(p["location"])


fig = plt.figure(figsize=(10,8))
for i in range(2):
    ax = plt.subplot(1,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(z_ts_6_4[0][47], cmap='gray', vmin=-0.00005, vmax=0.00005)
    if i%2==1:
        plt.scatter(beta_galactosidase_x, beta_galactosidase_y, edgecolor='red',facecolor='none')
    


thyroglobulin_x = []
thyroglobulin_y = []

file = open('/kaggle/input/czii-cryo-et-object-identification/train/overlay/ExperimentRuns/TS_6_4/Picks/thyroglobulin.json')
for p in json.loads(file.read())['points']:
    z=float(p["location"]["z"])
    if z >= 450 and z < 500 :
        thyroglobulin_x.append(float(p["location"]["x"])/10)
        thyroglobulin_y.append(float(p["location"]["y"])/10)
        print(p["location"])


fig = plt.figure(figsize=(10,8))
for i in range(2):
    ax = plt.subplot(1,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(z_ts_6_4[0][47], cmap='gray', vmin=-0.00005, vmax=0.00005)
    if i%2==1:
        plt.scatter(thyroglobulin_x, thyroglobulin_y, edgecolor='red',facecolor='none')
    


# {'x': 5274.903, 'y': 5288.121, 'z': 619.798}

fig = plt.figure(figsize=(10,2.5))

ax = plt.subplot(1,4,1)
plt.xticks([])
plt.yticks([])
plt.title('Original')
plt.imshow(z_ts_6_4[0][62],cmap="gray",vmin=-5e-5,vmax=5e-5)
plt.scatter([5274.903/10],[5288.121/10],edgecolor='red',facecolor='none')

ax = plt.subplot(1,4,2)
#plt.xticks([])
#plt.yticks([])
plt.title('Straight On')
plt.imshow(z_ts_6_4[0][62,513:543,512:542],cmap="gray")

ax = plt.subplot(1,4,3)
#plt.xticks([])
#plt.yticks([])
plt.title('Side view')
plt.imshow(np.transpose(z_ts_6_4[0],axes=(2,1,0))[527,507:547,42:82],cmap="gray")

ax = plt.subplot(1,4,4)
#plt.xticks([])
#plt.yticks([])
plt.title('Top view')
plt.imshow(np.transpose(z_ts_6_4[0],axes=(1,0,2))[528,42:82,508:548],cmap="gray")



#{'x': 911.29, 'y': 5638.402, 'z': 671.21}
#{'x': 5580.108, 'y': 1240.86, 'z': 692.222}
#{'x': 4765.58, 'y': 3469.964, 'z': 689.813}

fig = plt.figure(figsize=(10,2.5))

ax = plt.subplot(1,4,1)
#plt.xticks([])
#plt.yticks([])
plt.title('Original')
plt.imshow(z_ts_6_4[0][67],cmap="gray",vmin=-5e-5,vmax=5e-5)
plt.scatter([911.29/10],[5638.402/10],edgecolor='red',facecolor='none')

ax = plt.subplot(1,4,2)
#plt.xticks([])
#plt.yticks([])
plt.title('Straight On')
plt.imshow(z_ts_6_4[0][67,543:583,71:111],cmap="gray",vmin=-5e-5,vmax=5e-5)

ax = plt.subplot(1,4,3)
#plt.xticks([])
#plt.yticks([])
plt.title('Side view')
plt.imshow(np.transpose(z_ts_6_4[0],axes=(2,1,0))[91,533:573,47:87],cmap="gray",vmin=-5e-5,vmax=5e-5)

ax = plt.subplot(1,4,4)
#plt.xticks([])
#plt.yticks([])
plt.title('Top view')
plt.imshow(np.transpose(z_ts_6_4[0],axes=(1,0,2))[563,37:97,61:121],cmap="gray",vmin=-5e-5,vmax=5e-5)



# {'x': 1019.831, 'y': 1859.831, 'z': 400.424}

fig = plt.figure(figsize=(10,2.5))

ax = plt.subplot(1,4,1)
#plt.xticks([])
#plt.yticks([])
plt.title('Original')
plt.imshow(z_ts_6_4[0][40],cmap="gray",vmin=-5e-5,vmax=5e-5)
plt.scatter([1019.83/10],[1859.83/10],edgecolor='red',facecolor='none')

ax = plt.subplot(1,4,2)
#plt.xticks([])
#plt.yticks([])
plt.title('Straight On')
plt.imshow(z_ts_6_4[0][40,166:206,82:122],cmap="gray",vmin=-5e-5,vmax=5e-5)

ax = plt.subplot(1,4,3)
#plt.xticks([])
#plt.yticks([])
plt.title('Side view')
plt.imshow(np.transpose(z_ts_6_4[0],axes=(2,1,0))[102,166:206,20:60],cmap="gray",vmin=-5e-5,vmax=5e-5)

ax = plt.subplot(1,4,4)
#plt.xticks([])
#plt.yticks([])
plt.title('Top view')
plt.imshow(np.transpose(z_ts_6_4[0],axes=(1,0,2))[186,20:60,82:122],cmap="gray",vmin=-5e-5,vmax=5e-5)



# {'x': 804.615, 'y': 1977.846, 'z': 489.385}
# 2 1 0

fig = plt.figure(figsize=(10,2.5))

ax = plt.subplot(1,4,1)
#plt.xticks([])
#plt.yticks([])
plt.title('Original')
plt.imshow(z_ts_6_4[0][49],cmap="gray",vmin=-5e-5,vmax=5e-5)
plt.scatter([804.615/10],[1977.846/10],edgecolor='red',facecolor='none')

ax = plt.subplot(1,4,2)
#plt.xticks([])
#plt.yticks([])
plt.title('Straight On')
plt.imshow(z_ts_6_4[0][49,178:218,60:100],cmap="gray",vmin=-5e-5,vmax=5e-5)

ax = plt.subplot(1,4,3)
#plt.xticks([])
#plt.yticks([])
plt.title('Side view')
plt.imshow(np.transpose(z_ts_6_4[0],axes=(2,1,0))[80,178:218,29:69],cmap="gray",vmin=-5e-5,vmax=5e-5)

ax = plt.subplot(1,4,4)
#plt.xticks([])
#plt.yticks([])
plt.title('Top view')
plt.imshow(np.transpose(z_ts_6_4[0],axes=(1,0,2))[198,29:69,60:100],cmap="gray",vmin=-5e-5,vmax=5e-5)



# {'x': 5251.785, 'y': 2090.452, 'z': 490.516}
# 2 1 0

fig = plt.figure(figsize=(10,2.5))

ax = plt.subplot(1,4,1)
#plt.xticks([])
#plt.yticks([])
plt.title('Original')
plt.imshow(z_ts_6_4[0][49],cmap="gray",vmin=-5e-5,vmax=5e-5)
plt.scatter([5251/10],[2090/10],edgecolor='red',facecolor='none')

ax = plt.subplot(1,4,2)
#plt.xticks([])
#plt.yticks([])
plt.title('Straight On')
plt.imshow(z_ts_6_4[0][49,189:228,505:545],cmap="gray",vmin=-5e-5,vmax=5e-5)

ax = plt.subplot(1,4,3)
#plt.xticks([])
#plt.yticks([])
plt.title('Side view')
plt.imshow(np.transpose(z_ts_6_4[0],axes=(2,1,0))[525,189:228,29:69],cmap="gray",vmin=-5e-5,vmax=5e-5)

ax = plt.subplot(1,4,4)
#plt.xticks([])
#plt.yticks([])
plt.title('Top view')
plt.imshow(np.transpose(z_ts_6_4[0],axes=(1,0,2))[209,29:69,505:545],cmap="gray",vmin=-5e-5,vmax=5e-5)



fig = plt.figure(figsize=(10,64))
for i in range(158):
    ax = plt.subplot(18,9,i+1)
    if i != 0:
        plt.xticks([])
        plt.yticks([])
    plt.imshow(np.transpose(z_ts_6_4[2],axes=(2,1,0))[i])


fig = plt.figure(figsize=(10,50))
for i in range(158):
    ax = plt.subplot(40,4,i+1)
    if i != 0:
        plt.xticks([])
        plt.yticks([])
    plt.imshow(np.transpose(z_ts_6_4[2],axes=(1,0,2))[i])


circle = plt.Circle((15,15),radius=10,edgecolor='r',facecolor="none")
plt.gca().add_patch(circle)
plt.imshow(z_ts_6_4[0][62,513:543,512:542],cmap="gray")



circle = plt.Circle((20,20),radius=15,edgecolor='r',facecolor="none")
plt.gca().add_patch(circle)
plt.imshow(np.transpose(z_ts_6_4[0],axes=(2,1,0))[527,507:547,42:82],cmap="gray")



circle = plt.Circle((20,20),radius=15,edgecolor='r',facecolor="none")
plt.gca().add_patch(circle)
plt.imshow(np.transpose(z_ts_6_4[0],axes=(1,0,2))[528,42:82,508:548],cmap="gray")





