# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


        if choice == '1':
            display_tasks(tasks)
        elif choice == '2':
            desc = input("Enter task description: ").strip()
            if desc:
                add_task(tasks, desc)
            else:
                print("Description cannot be empty.")
        elif choice == '3':
            display_tasks(tasks)
            try:
                idx = int(input("Enter task number to complete: "))
                complete_task(tasks, idx)
            except ValueError:
                print("Please enter a valid number.")
        elif choice == '4':
            display_tasks(tasks)
            try:
                idx = int(input("Enter task number to delete: "))
                delete_task(tasks, idx)
            except ValueError:
                print("Please enter a valid number.")
        elif choice == '5':
            save_tasks(tasks)
            print("Tasks saved. Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")
        
        save_tasks(tasks)  # Auto-save after each action
if __name__ == "__main__":
    main()

