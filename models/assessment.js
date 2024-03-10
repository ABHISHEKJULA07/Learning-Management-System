const tf = require('@tensorflow/tfjs');

function correctAssessment(studentAnswers) {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [assessments.length], units: 10, activation: 'relu' }),
      tf.layers.dense({ units: 1, activation: 'linear' }),
    ],
  });

  const studentAnswersTensor = tf.tensor2d([studentAnswers]);
  const correctedScore = model.predict(studentAnswersTensor).dataSync()[0];
  const maxPossibleScore = assessments.reduce((acc, assessment) => acc + assessment.questions, 0);
  const correctedPercentage = Math.min(100, Math.max(0, correctedScore)) / maxPossibleScore;

  return {
    correctedScore: correctedScore,
    correctedPercentage: correctedPercentage * 100,
  };
}


export default correctAssessment;
