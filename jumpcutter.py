import subprocess
import math
import os
import argparse
import numpy as np
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter    
from scipy.io import wavfile


def getMaxVolume(s):
    return max(float(np.max(s)), -float(np.min(s)))

def copyFrame(inputFrame, outputFrame, list):
    src = "TMP/{}.jpg".format(inputFrame + 1)
    if not os.path.isfile(src):
        return False
    list.append("file '{}'\n".format(src))
    if outputFrame%500 == 499:
        print(str(outputFrame + 1) + " time-altered frames saved.")
    return True

def inputToOutputFilename(filename):
    dotIndex = filename.rfind(".")
    return filename[:dotIndex] + "_ALTERED" + filename[dotIndex:]

def deletePath():
    subprocess.call('rm -rf TMP', shell=True)

def createPath(s):
    try:  
        os.mkdir('TMP')
    except OSError:
        a = input('Do you want to delete the TMP folder? [y/N]')
        if a == 'y':
            subprocess.call('rm -rf TMP', shell=True)
            os.mkdir('TMP')
            return 
        assert False, "Creation of the directory %s failed. (The TEMP folder may already exist. Delete or rename it, and try again.)"


parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str,  help="the input file")
parser.add_argument('-o', type=str, default="", help="the output file")
parser.add_argument('--st', type=float, default=0.04, help="the volume amount that frames' audio needs to surpass to be consider \"sounded\"")
parser.add_argument('--sound', type=float, default=1.15, help="the speed that sounded (spoken) frames should be played at.")
parser.add_argument('--silent', type=float, default=6.00, help="the speed that silent frames should be played at")
parser.add_argument('--fm', type=float, default=1, help="some silent frames adjacent to sounded frames are included to provide context")
parser.add_argument('--sr', type=float, default=44100, help="sample rate of the input and output videos")
parser.add_argument('-r', type=float, default=30, help="frame rate of the input and output videos.")
parser.add_argument('-q', type=int, default=4, help="quality of frames to be extracted from input video.")

createPath('TMP')

args = parser.parse_args()

frameRate = args.r
SAMPLE_RATE = args.sr
SILENT_THRESHOLD = args.st
FRAME_SPREADAGE = args.fm
NEW_SPEED = [args.silent, args.sound]
INPUT_FILE = args.i
FRAME_QUALITY = args.q

assert INPUT_FILE != None , "What am I supposed to do ?!"

OUTPUT_FILE = args.o if len(args.o) >= 1 else inputToOutputFilename(INPUT_FILE)

AUDIO_FADE_ENVELOPE_SIZE = 400
    


command = "ffmpeg -i '{}' -qscale:v {} TMP/%d.jpg -hide_banner".format(INPUT_FILE, FRAME_QUALITY)
subprocess.call(command, shell=True)

command = "ffmpeg -i '{}' -ab 160k -ac 2 -ar '{}' -vn TMP/audio.wav".format(INPUT_FILE, SAMPLE_RATE)
subprocess.call(command, shell=True)

sampleRate, audioData = wavfile.read('TMP/audio.wav')
audioSampleCount = audioData.shape[0]
maxAudioVolume = getMaxVolume(audioData)


samplesPerFrame = sampleRate/frameRate
audioFrameCount = int(math.ceil(audioSampleCount/samplesPerFrame))
hasLoudAudio = np.zeros((audioFrameCount))

for i in range(audioFrameCount):
    start = int(i * samplesPerFrame)
    end = min(int((i+1) * samplesPerFrame), audioSampleCount)
    audiochunks = audioData[start:end]
    maxchunksVolume = float(getMaxVolume(audiochunks))/maxAudioVolume
    if maxchunksVolume >= SILENT_THRESHOLD:
        hasLoudAudio[i] = 1

chunks = [[0,0,0]]
shouldIncludeFrame = np.zeros((audioFrameCount))
for i in range(audioFrameCount):
    start = int(max(0 ,i - FRAME_SPREADAGE))
    end = int(min(audioFrameCount, i + 1 + FRAME_SPREADAGE))
    shouldIncludeFrame[i] = np.max(hasLoudAudio[start:end])
    if (i >= 1 and shouldIncludeFrame[i] != shouldIncludeFrame[i-1]):
        chunks.append([chunks[-1][1], i, shouldIncludeFrame[i-1]])

chunks.append([chunks[-1][1], audioFrameCount, shouldIncludeFrame[i-1]])
chunks = chunks[1:]

outputAudioData = np.zeros((0,audioData.shape[1]))
outputPointer = 0

lastExistingFrame = None
fileFrame = open("fileFrame.txt", "w")
sFile = 'TMP/tempStart.wav'
eFile = 'TMP/tempEnd.wav'
for chunk in chunks:
    audioChunk = audioData[int(chunk[0]*samplesPerFrame) : int(chunk[1]*samplesPerFrame)]
    
    wavfile.write(sFile, SAMPLE_RATE, audioChunk)
    with WavReader(sFile) as reader:
        with WavWriter(eFile, reader.channels, reader.samplerate) as writer:
            tsm = phasevocoder(reader.channels, speed=NEW_SPEED[int(chunk[2])])
            tsm.run(reader, writer)
    _, alteredAudioData = wavfile.read(eFile)
    leng = alteredAudioData.shape[0]
    endPointer = outputPointer + leng
    outputAudioData = np.concatenate((outputAudioData, alteredAudioData/maxAudioVolume))

    if leng < AUDIO_FADE_ENVELOPE_SIZE:
        outputAudioData[outputPointer:endPointer] = 0
    else:
        premask = np.arange(AUDIO_FADE_ENVELOPE_SIZE)/AUDIO_FADE_ENVELOPE_SIZE
        mask = np.repeat(premask[:, np.newaxis], 2, axis=1)
        outputAudioData[outputPointer:outputPointer + AUDIO_FADE_ENVELOPE_SIZE] *= mask
        outputAudioData[endPointer - AUDIO_FADE_ENVELOPE_SIZE:endPointer] *= 1-mask
    
    
    startOutputFrame = int(math.ceil(outputPointer/samplesPerFrame))
    endOutputFrame = int(math.ceil(endPointer/samplesPerFrame))
    frameList = []
    for outputFrame in range(startOutputFrame, endOutputFrame):
        inputFrame = int(chunk[0] + NEW_SPEED[int(chunk[2])] * (outputFrame-startOutputFrame))
        didItWork = copyFrame(inputFrame, outputFrame, frameList)
        if didItWork:
            lastExistingFrame = inputFrame
        elif lastExistingFrame != None:
            copyFrame(lastExistingFrame, outputFrame, frameList)
    
    fileFrame.write(''.join(frameList))
    outputPointer = endPointer
fileFrame.close()

try:
    wavfile.write('TMP/audioNew.wav', SAMPLE_RATE, outputAudioData)
except:
    try:
        wavfile.write('TMP/audioNew.wav', 44056, outputAudioData)
    except:
        try:
            wavfile.write('TMP/audioNew.wav', 37800, outputAudioData)
        except:
            wavfile.write('TMP/audioNew.wav', 32000 , outputAudioData)

command = "ffmpeg -r {} -f concat -safe 0 -i fileFrame.txt -i TMP/audioNew.wav -r {} '{}'".format(frameRate, frameRate, OUTPUT_FILE)
subprocess.call(command, shell=True)

deletePath()
subprocess.call('rm fileFrame.txt', shell=True)
