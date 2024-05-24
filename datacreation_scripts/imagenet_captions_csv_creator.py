import pandas as pd
import json

with open("/home/nas2_userH/hyesulim/Data/imagenet_captions.json", "r") as f:
    captions = json.load(f)

# convert data into dataframe
captions = pd.DataFrame(captions)

# creat caption by combining title, tags, description
captions["caption"] = (
    captions["title"]
    + " "
    + pd.Series(list(map(" ".join, captions["tags"])))
    + " "
    + captions["description"]
)
captions.head()

# delete column title, tags, description, wnid, filename
captions.drop(
    columns=["title", "tags", "description", "wnid", "filename"], inplace=True
)

# change column name from caption to title
captions.rename(columns={"caption": "title"}, inplace=True)

# replace '\r' with ' ' in title
captions["title"] = captions["title"].str.replace("\r", " ")

# save csv
captions.to_csv("./datasets/csv/imagenet_captions.csv", sep="\t", index=False)
