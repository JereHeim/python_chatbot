import React from 'react';

function TrainingChecker() {
  // Functions should be inside the component or imported from elsewhere

  const checkAnswers = () => {
    const inputs = document.querySelectorAll('#training-code input.blank-input');
    inputs.forEach(input => {
      const expected = input.dataset.answer.trim();
      const actual = input.value.trim();
      if (actual === expected) {
        input.style.borderColor = 'green';
        input.style.backgroundColor = '#d4edda';
      } else {
        input.style.borderColor = 'red';
        input.style.backgroundColor = '#f8d7da';
      }
    });
  };

  const resetInputs = () => {
    const inputs = document.querySelectorAll('#training-code input.blank-input');
    inputs.forEach(input => {
      input.value = '';
      input.style.borderColor = '';
      input.style.backgroundColor = '';
    });
  };

  return (
    <>
      {/* Your JSX here, e.g. buttons to trigger these functions */}
      <button onClick={checkAnswers}>Check Answers</button>
      <button onClick={resetInputs}>Reset</button>
    </>
  );
}

export default TrainingChecker;
// This file is a React component that provides functionality to check answers and reset inputs for coding training exercises.
// It includes functions to validate user input against expected answers and visually indicate correctness with styles.