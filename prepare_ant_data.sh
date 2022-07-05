

# download all data to ../data/ants/
# root
#   --data
#      --ant
#         --userTrain_videos
#         --submission_videos
#         ...
#   --code

mkdir -p ../data/ant && cd ../data/ant
unzip userTrain_videos.zip ./userTrain_videos
unzip submission_videos.zip ./submission_videos
mkdir video_clips_512
ln -s ./userTrain_videos/* ./video_clips_512/
ln -s ./submission_videos/* ./video_clips_512/
