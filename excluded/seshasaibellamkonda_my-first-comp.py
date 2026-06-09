import pandas as pd


food = pd.read_csv("/kaggle/input/messy-food-waste-prediction-dataset/test.csv", parse_dates=["date"],date_format="%Y-%m-%d")


food.head()


food.info()





food.info()


col_new_names = {
    "date": "Date",
    "meals_served": "Meals_Served",
    "kitchen_staff": "Kitchen_Staff",
    "temperature_C": "Temperature_Celsius",
    "humidity_percent": "Humidity_Percent",
    "day_of_week": "Day_of_Week",
    "special_event": "Special_Event",
    "past_waste_kg": "Past_Waste_Kg",
    "staff_experience": "Staff_Experience",
    "waste_category": "Waste_Category"
}
food = food.rename(columns= col_new_names)


food.head()


food["Staff_Experience"].mode()


food["Staff_Experience"] = food["Staff_Experience"].fillna("Beginner")


food["Staff_Experience"].unique()


food["Staff_Experience"] = food["Staff_Experience"].replace("intermediate","Intermediate")


food.head()


food["Waste_Category"].unique()


new_Waste_Categories = {
    "dairy": "Dairy",
    "MeAt": "Meat",
    "MEAT": "Meat",
    "Vegetables": "Vegetables",
    "GRAINS": "Grains"
}
food["Waste_Category"] = food["Waste_Category"].replace(new_Waste_Categories)


food["Waste_Category"].unique()


food["Waste_Category"] =food["Waste_Category"].astype("category")
food["Staff_Experience"] = food["Staff_Experience"].astype("category")

food.info()



food["Meals_Served"]





temp_min , temp_high = 10,45
food.loc[(food["Temperature_Celsius"] < temp_min) | (food["Temperature_Celsius"] > temp_high),"Temperature_Celsius"] = None
food["Temperature_Celsius"] = food["Temperature_Celsius"].fillna(food["Temperature_Celsius"].median())


food["Meals_Served"].describe()


max_meal_ser = 600
food.loc[(food["Meals_Served"]> max_meal_ser),"Meals_Served"]= None
food["Meals_Served"] = food["Meals_Served"].fillna(food["Meals_Served"].mean())


food.info()


food = food.drop_duplicates()
food.reset_index(drop = True,inplace = True)


food.info()


food.to_csv("/kaggle/working/final_submission.csv", index=False)







