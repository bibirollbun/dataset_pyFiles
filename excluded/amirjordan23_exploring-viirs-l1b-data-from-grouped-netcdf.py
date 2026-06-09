!pip install netCDF4


import google.generativeai as genai
import os
# from google import genai
import netCDF4
import xarray as xr
import matplotlib.pyplot as plt


# client = genai.Client(api_key="Gemini_api")

# prompt = """Produce a detailed plan for a research scientist and provide recommendations 
#     about how and where they could use Gemini models to analyze multi-spectral satellite imagery 
#     with the goal of discovering evidence of ancient civilizations in Brazil."""


# response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     contents=prompt,
# )

# print(response.text)



# for m in genai.list_models():
#   if 'generateContent' in m.supported_generation_methods:
#     print(m.name)


try:
    genai.configure(api_key="gemini_api")
except AttributeError:
    print("Error: The 'genai.configure' function was not found. Make sure you have the correct library version.")
    print("Consider using 'genai.Client(api_key=...) if you are on an older version or a different setup.")
    exit()



try:
    model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17') # Or 'gemini-1.5-flash', 'gemini-1.0-pro', etc.
except Exception as e:
    print(f"Error initializing the model: {e}")
    exit()


prompt = """Produce a detailed plan for a research scientist and provide recommendations 
    about how and where they could use Gemini models to analyze multi-spectral satellite imagery 
    with the goal of discovering evidence of ancient civilizations in Brazil."""

# try:
#     response = model.generate_content(prompt)
#     print(response.text)

    # For streaming responses (useful for longer content):
    # response_stream = model.generate_content(prompt, stream=True)
    # for chunk in response_stream:
    #   print(chunk.text)

# except Exception as e:
#     print(f"An error occurred during content generation: {e}")


# Open the NetCDF file
file_path = '/kaggle/input/lancemodis2598344475/VJ102MOD_NRT.A2025144.0406.021.2025144060649.nc'
dataset = netCDF4.Dataset(file_path, 'r') # 'r' is for read mode


print("Dimensions:")
for dim_name, dim_obj in dataset.dimensions.items():
    print(f"  {dim_name}: {len(dim_obj)}")


print("\nGlobal attributes:")
for attr_name in dataset.ncattrs():
    print(f"  {attr_name}: {dataset.getncattr(attr_name)}")


print("\nVariables:")
for var_name, var_obj in dataset.variables.items():
    print(f"  Name: {var_name}")
    print(f"    Dimensions: {var_obj.dimensions}")
    print(f"    Shape: {var_obj.shape}")
    print(f"    Units: {var_obj.units if 'units' in var_obj.ncattrs() else 'N/A'}")
    # To get actual data (use slicing for large datasets):
    # data = var_obj[:]


dataset.close()


FILE_PATH = '/kaggle/input/lancemodis2598344475/VJ102MOD_NRT.A2025144.0406.021.2025144060649.nc'


GROUP_PATH = 'observation_data' 

# CHOOSE A VARIABLE TO PLOT from the 'observation_data' group.
# Since your file is 'Night' data (DayNightFlag: Night from earlier analysis),
# thermal bands are most interesting. Good choices would be:
# 'M12', 'M13', 'M14', 'M15', 'M16'
# M07, M08, M10, M11 are typically reflected solar bands and will likely be dark.
VARIABLE_TO_PLOT = 'M16'


def view_nc_image_from_group(file_path, group_path, variable_name):
    """
    Opens a specific group in a NetCDF file and plots a 2D variable as an image.
    """
    try:
        
        ds_group = xr.open_dataset(file_path, group=group_path)
        print(f"Successfully opened group '{group_path}' in dataset: {file_path}\n")

        print(f"--- Variables available in Group '{group_path}' ---")
        if ds_group.data_vars:
            for var_name_in_group in ds_group.data_vars:
                print(f"- {var_name_in_group} (Dimensions: {ds_group[var_name_in_group].dims}, Shape: {ds_group[var_name_in_group].shape})")
        else:
            print(f"No data variables found in group '{group_path}'.")
            ds_group.close()
            return
        print("--------------------------------------------------\n")

        if variable_name not in ds_group:
            print(f"ERROR: Variable '{variable_name}' not found in group '{group_path}'.")
            print("Please choose one of the variables listed above for this group and update VARIABLE_TO_PLOT.")
            ds_group.close()
            return

        
        data_array = ds_group[variable_name]

        print(f"Selected variable: '{variable_name}' from group '{group_path}'")
        print(f"Dimensions: {data_array.dims}")
        print(f"Shape: {data_array.shape}")
        if 'long_name' in data_array.attrs: # Check if attribute exists
            print(f"Long Name: {data_array.attrs['long_name']}")
        if 'units' in data_array.attrs: # Check if attribute exists
            print(f"Units: {data_array.attrs['units']}")
        print("-" * 30)

        
        if data_array.ndim == 2:
            plottable_data = data_array
        elif data_array.ndim > 2: # Should not be needed for these M-bands, but good practice
            first_dim_name = data_array.dims[0]
            plottable_data = data_array.isel({first_dim_name: 0})
            print(f"NOTE: Data has more than 2 dimensions. Plotting the first slice along dimension '{first_dim_name}'.")
            print(f"Original shape: {data_array.shape}, Plotted shape: {plottable_data.shape}")
        else:
            print(f"ERROR: Variable '{variable_name}' is not 2D or more, cannot plot as a simple image. Shape: {data_array.shape}")
            ds_group.close()
            return

        # Plot the data
        plt.figure(figsize=(12, 9)) 
        
        plottable_data.plot(cmap='inferno', robust=True) 
        
        title = f"Visualization of: {variable_name} (from group '{group_path}')"
        if 'long_name' in data_array.attrs:
            title += f"\n({data_array.attrs['long_name']})"
        plt.title(title)
        
        plt.xlabel(f"Dimension: {data_array.dims[1]}") # Typically pixels
        plt.ylabel(f"Dimension: {data_array.dims[0]}") # Typically lines/scans
        
        plt.tight_layout()
        plt.show()

        
        ds_group.close()

    except FileNotFoundError:
        print(f"ERROR: File not found at '{file_path}'. Please check the path and filename.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
        if 'group not found' in str(e).lower() or 'HDF5ExtError' in str(e): 
            print(f"Detail: It seems the group '{group_path}' might not exist or there's an HDF5 issue opening it.")


if __name__ == '__main__':
    view_nc_image_from_group(FILE_PATH, GROUP_PATH, VARIABLE_TO_PLOT)




