#!/bin/bash
# A script to generate an ALE video with FFMpeg, *nix systems. 

# -r ## specifies the frame rate
# -i record/%06d.png indicates we should use sequentially numbered frames in directory 'record'
# -i sound.wav indicates the location of the sound file
# -f mov specifies a MOV format
# -c:a mp3 specifies the sound codec 
# -c:v libx264 specifies the video codec
# 


# Attempt to use ffmpeg. If this fails, use avconv (fix for Ubuntu 14.04). 
{
    ffmpeg -r 60 -i record/%06d.png -i record/sound.wav -f mov -c:a mp3 -c:v libx264 agent.mov
} || {
    avconv -r 60 -i record/%06d.png -i record/sound.wav -f mov -c:a mp3 -c:v libx264 agent.mov
}

