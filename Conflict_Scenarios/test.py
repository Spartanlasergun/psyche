import pandas as pd

# Read Data From Archive
print("Reading Data...")
data = pd.read_csv('Conflict Scenarios Research.csv')
# create documents list
documents = data["Describe a past experience you've had that involved conflict with a family member, friend, or significant other. Be as detailed as you like."].tolist()
print("Number of Documents: " + str(len(documents)))

count = 0
for item in documents:
    if isinstance(item, str):
        count = count + 1

print(count)