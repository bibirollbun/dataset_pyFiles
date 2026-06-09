%%writefile resize_videos.sh
#!/bin/bash

# arguments check
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_folder> <output_folder> <resolution>"
    echo "Example: $0 ./input ./output 1920x1080"
    exit 1
fi

# inputs 
INPUT_FOLDER=$1
OUTPUT_FOLDER=$2
RESOLUTION=$3

# file check
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder '$INPUT_FOLDER' does not exist."
    exit 1
fi

# create folder
mkdir -p "$OUTPUT_FOLDER"

# get width and height (must be even numbers)
WIDTH=$(echo "$RESOLUTION" | cut -d'x' -f1)
HEIGHT=$(echo "$RESOLUTION" | cut -d'x' -f2)

# loop all .mp4 files
for input_file in "$INPUT_FOLDER"/*.mp4; do
    
    if [ ! -e "$input_file" ]; then
        echo "No .mp4 files found in '$INPUT_FOLDER'."
        exit 1
    fi

    filename=$(basename -- "$input_file")
    filename_no_ext="${filename%.*}"

    output_file="$OUTPUT_FOLDER/${filename_no_ext}_${RESOLUTION}.mp4"

    # 调用 ffmpeg 进行视频大小调整
    echo "Resizing $input_file to $RESOLUTION..."
    ffmpeg -i "$input_file" \
           -vf "scale=$WIDTH:$HEIGHT" \
           -loglevel error \
           -c:a copy \
           "$output_file"

    if [ $? -eq 0 ]; then
        echo "Successfully resized $input_file to $output_file"
    else
        echo "Failed to resize $input_file"
    fi
done

echo "All videos processed."


!chmod +x resize_videos.sh


!./resize_videos.sh /kaggle/input/nexar-collision-prediction/train train_resized 224x224


!./resize_videos.sh /kaggle/input/nexar-collision-prediction/test test_resized 224x224


from IPython.display import Video
import glob
import random

videos_resized = glob.glob("train_resized/*.mp4")
print(len(videos_resized))


Video(videos_resized[random.randint(0, len(videos_resized))], embed=True)


videos_resized = glob.glob("test_resized/*.mp4")
print(len(videos_resized))


Video(videos_resized[random.randint(0, len(videos_resized))], embed=True)




