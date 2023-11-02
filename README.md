# ASL Fingerspelling Translator
American Sign Language is one of the more common types of sign language used by many deaf or hard-of-hearing people around the world. Unfortunately, despite this language being so popular it is still quite rare for people to be able to perform ASL, even the parents of deaf/HOH children. As such, my team decided to build a simplistic translator for non-speakers so that they understand fingerspelt words and be shown how to spell words back for a university assignment (UTS Software Innovation Studio, Autumn 2023). 

The concept my team (Team 16) came up with was to develop a mixed reality/augmented reality application that would detect ASL letters and perform live subtitling (and autocorrection) for the user. As an extension we added English->ASL so that the user would also be able to spell words and phrases back to the ASL speaker. 

## Repository
This repository contains the dataset builder and model notebook that was used to construct the recognition/translation model that perform symbol to letter conversions.

### Dataset Construction
This is stored in `data/`

#### Dataset
I have not publicised my dataset as it is not very well constructed since it is just me holding up fingerspelling letters. I would recommend you create your own for each letter. This can be done with a simple frame dump of a video. I would recommend at least 30,000 images.

The dataset consists of landmarks collected by MediaPipe. Currently the collected landmarks are:
* Hand Landmarks: `range(0, 21)`
* Pose Landmarks: None
* Face Landmarks: None

### Model
The model notebook is `cnn_model.ipynb`. A TFLite export can be found in the releases section of this repository. 

## Credits
Lachlan Wright -> Helped develop autocorrect functionality that makes use of an NLP
Lachlan Garrity -> Developed MR/AR interface
Brendan Khavin -> Helped develop autocorrect functionality, text to speech, and assess model structure viabilities during the early phases.

### Original Github Organisation and Repos
**Organisation**: [SIS Team 16](https://github.com/2023SIS-Team16) <br/>
**Original Model Repository**: [CV API](https://github.com/2023SIS-Team16/cv_api) <br/>
**Autocorrect/NLP Repository**: [NLP API](https://github.com/2023SIS-Team16/lachlanw_llama) <br/>
**MR/AR Unity Project**: [Unity Project](https://github.com/2023SIS-Team16/SIS_UnityProject) <br/>