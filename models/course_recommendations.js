const tf = require("@tensorflow/tfjs");

export function contentBasedFiltering(userPreferences) {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({
        inputShape: [courses.length],
        units: 10,
        activation: "relu",
      }),
      tf.layers.dense({ units: courses.length, activation: "softmax" }),
    ],
  });

  const userPrefs = tf.tensor2d([userPreferences]);
  const predictions = model.predict(userPrefs);
  const predictedCourses = predictions.dataSync();

  const recommendedCourses = predictedCourses.map((score, index) => ({
    course: courses[index],
    score: score,
  }));

  recommendedCourses.sort((a, b) => b.score - a.score);

  return recommendedCourses;
}
