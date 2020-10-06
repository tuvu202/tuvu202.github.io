/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as posenet from '@tensorflow-models/posenet';
import * as tf from '@tensorflow/tfjs';
import { calculateAngle, drawBoundingBox, drawKeypoints, drawSkeleton, isMobile, toggleLoadingUI } from './utils';

const videoWidth = window.innerWidth * 0.7;
const videoHeight = window.innerHeight;

/**
 * Loads a the camera to be used in the demo
 */
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error('Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}

const defaultQuantBytes = 2;
const defaultResNetMultiplier = 1.0;
const defaultResNetStride = 32;
const defaultResNetInputResolution = 250;

const guiState = {
  algorithm: 'single-pose',
  input: {
    architecture: 'ResNet50',
    outputStride: defaultResNetStride,
    inputResolution: defaultResNetInputResolution,
    multiplier: defaultResNetMultiplier,
    quantBytes: defaultQuantBytes,
  },
  singlePoseDetection: {
    minPoseConfidence: 0.7,
    minPartConfidence: 0.7,
  },
  multiPoseDetection: {
    maxPoseDetections: 5,
    minPoseConfidence: 0.15,
    minPartConfidence: 0.1,
    nmsRadius: 30.0,
  },
  output: {
    showVideo: true,
    showSkeleton: true,
    showPoints: true,
    showBoundingBox: false,
  },
  net: null,
  image: 'Vs9p4Sj.jpg',
};
const imageBucket = 'https://i.imgur.com/';
const distance = 20;
let userResult = {
  leftElbow: 0,
};
let imgResult = {
  leftElbow: 0,
};

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video, net) {
  const canvas = document.getElementById('output');
  const ctx = canvas.getContext('2d');

  // since images are being fed from a webcam, we want to feed in the
  // original image and then just flip the keypoints' x coordinates. If instead
  // we flip the image, then correcting left-right keypoint pairs requires a
  // permutation on all the keypoints.
  const flipPoseHorizontal = true;

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  async function poseDetectionFrame() {
    let poses = [];
    let minPoseConfidence;
    let minPartConfidence;
    switch (guiState.algorithm) {
      case 'single-pose':
        const pose = await guiState.net.estimatePoses(video, {
          flipHorizontal: flipPoseHorizontal,
          decodingMethod: 'single-person',
        });
        poses = poses.concat(pose);
        minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
        break;
      case 'multi-pose':
        let allPoses = await guiState.net.estimatePoses(video, {
          flipHorizontal: flipPoseHorizontal,
          decodingMethod: 'multi-person',
          maxDetections: guiState.multiPoseDetection.maxPoseDetections,
          scoreThreshold: guiState.multiPoseDetection.minPartConfidence,
          nmsRadius: guiState.multiPoseDetection.nmsRadius,
        });

        poses = poses.concat(allPoses);
        minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;
        break;
    }

    ctx.clearRect(0, 0, videoWidth, videoHeight);

    if (guiState.output.showVideo) {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-videoWidth, 0);
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      ctx.restore();
    }

    // For each pose (i.e. person) detected in an image, loop through the poses
    // and draw the resulting skeleton and keypoints if over certain confdence   // scoes    poses.forEac(({ score, keypoints }) => {
    poses.forEach(({ score, keypoints }) => {
      if (score >= minPoseConfidence) {
        userResult.leftElbow = calculateAngle(keypoints[9], keypoints[7], keypoints[5]);
        recalculate();
        if (guiState.output.showPoints) {
          drawKeypoints(keypoints, minPartConfidence, ctx);
        }
        if (guiState.output.showSkeleton) {
          drawSkeleton(keypoints, minPartConfidence, ctx);
        }
        if (guiState.output.showBoundingBox) {
          drawBoundingBox(keypoints, ctx);
        }
      }
    });

    requestAnimationFrame(poseDetectionFrame);
  }

  poseDetectionFrame();
}

async function loadImage(imagePath) {
  const image = new Image();
  const promise = new Promise((resolve, reject) => {
    image.crossOrigin = '';
    image.onload = () => {
      resolve(image);
    };
  });

  image.src = `${imageBucket}${imagePath}`;
  return promise;
}

let image = null;
let predictedPoses = null;

/**
 * Purges variables and frees up GPU memory using dispose() method
 */
function disposePoses() {
  if (predictedPoses) {
    predictedPoses = null;
  }
}

async function testImageAndEstimatePoses(net) {
  // Purge prevoius variables and free up GPU memory
  disposePoses();

  // Load an example image
  image = await loadImage(guiState.image);

  // Creates a tensor from an image
  const input = tf.browser.fromPixels(image);

  // Estimates poses
  const poses = await net.estimatePoses(input, {
    flipHorizontal: false,
    decodingMethod: 'single-person',
    maxDetections: guiState.multiPoseDetection.maxDetections,
    scoreThreshold: guiState.multiPoseDetection.minPartConfidence,
    nmsRadius: guiState.multiPoseDetection.nmsRadius,
  });
  predictedPoses = poses[0];

  if (predictedPoses.score >= guiState.singlePoseDetection.minPoseConfidence) {
    imgResult.leftElbow = calculateAngle(predictedPoses.keypoints[9], predictedPoses.keypoints[7], predictedPoses.keypoints[5]);
  }

  input.dispose();
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
 */
export async function bindPage() {
  toggleLoadingUI(true);
  const net = await posenet.load({
    architecture: guiState.input.architecture,
    outputStride: guiState.input.outputStride,
    inputResolution: guiState.input.inputResolution,
    multiplier: guiState.input.multiplier,
    quantBytes: guiState.input.quantBytes,
  });
  toggleLoadingUI(false);
  let video;

  try {
    video = await loadVideo();
    toggleLoadingUI(false);
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' + 'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }
  guiState.net = net;
  detectPoseInRealTime(video, net);
  testImageAndEstimatePoses(guiState.net);
}

const recalculate = () => {
  if (userResult.leftElbow && Math.abs(imgResult.leftElbow - userResult.leftElbow) <= distance) {
    const percent = (1 - Math.abs(imgResult.leftElbow - userResult.leftElbow) / distance) * 100;
    let score = document.getElementById('score-placeholder');
    score.innerHTML = 'SCORE' + ': ' + Math.round(percent);
  }
};

navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();
