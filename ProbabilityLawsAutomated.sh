#!/bin/bash

# Function to calculate the Addition Law of Probability
addition_law() {
  echo "Addition Law of Probability: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)"
  echo "Please enter the probability of event A (P(A)): "
  read P_A
  echo "Please enter the probability of event B (P(B)): "
  read P_B
  echo "Please enter the probability of both events A and B happening (P(A ∩ B)): "
  read P_A_and_B
  result=$(echo "$P_A + $P_B - $P_A_and_B" | bc -l)
  echo "P(A ∪ B) = $result"
}

# Function to calculate the Multiplication Law of Probability
multiplication_law() {
  echo "Multiplication Law of Probability: P(A ∩ B) = P(A) * P(B|A)"
  echo "Please enter the probability of event A (P(A)): "
  read P_A
  echo "Please enter the probability of event B given A (P(B|A)): "
  read P_B_given_A
  result=$(echo "$P_A * $P_B_given_A" | bc -l)
  echo "P(A ∩ B) = $result"
}

# Function to calculate the Complement Law
complement_law() {
  echo "Complement Law: P(A') = 1 - P(A)"
  echo "Please enter the probability of event A (P(A)): "
  read P_A
  result=$(echo "1 - $P_A" | bc -l)
  echo "P(A') = $result"
}

# Function to calculate Conditional Probability
conditional_probability() {
  echo "Conditional Probability: P(A|B) = P(A ∩ B) / P(B)"
  echo "Please enter the probability of both events A and B happening (P(A ∩ B)): "
  read P_A_and_B
  echo "Please enter the probability of event B (P(B)): "
  read P_B
  result=$(echo "$P_A_and_B / $P_B" | bc -l)
  echo "P(A|B) = $result"
}

# Function to calculate Bayes' Theorem
bayes_theorem() {
  echo "Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)"
  echo "Please enter the probability of B given A (P(B|A)): "
  read P_B_given_A
  echo "Please enter the probability of event A (P(A)): "
  read P_A
  echo "Please enter the probability of event B (P(B)): "
  read P_B
  result=$(echo "($P_B_given_A * $P_A) / $P_B" | bc -l)
  echo "P(A|B) = $result"
}

# Function to calculate the Law of Total Probability
total_probability() {
  echo "Law of Total Probability: P(A) = Σ P(A|B_i) * P(B_i)"
  echo "Please enter the number of partitions of the sample space: "
  read n
  total_probability=0
  for ((i=1; i<=n; i++)); do
    echo "Enter the probability of event A given partition B$i (P(A|B$i)): "
    read P_A_given_B
    echo "Enter the probability of partition B$i (P(B$i)): "
    read P_B_i
    total_probability=$(echo "$total_probability + ($P_A_given_B * $P_B_i)" | bc -l)
  done
  echo "P(A) = $total_probability"
}

# Function to calculate the Weak Law of Large Numbers
weak_law_large_numbers() {
  echo "Weak Law of Large Numbers: The sample mean will converge to the population mean as sample size increases."
  echo "This law doesn't involve a calculation, but it describes how averages stabilize as sample size increases."
}

# Function to calculate the Central Limit Theorem
central_limit_theorem() {
  echo "Central Limit Theorem: The distribution of sample means will approach a normal distribution as sample size increases."
  echo "This law doesn't require input for a calculation but informs you about the distribution shape as sample size grows."
}

# Function to calculate the Markov's Inequality
markov_inequality() {
  echo "Markov's Inequality: P(X ≥ a) ≤ E[X] / a"
  echo "Please enter the expected value of X (E[X]): "
  read E_X
  echo "Please enter the value of a: "
  read a
  result=$(echo "$E_X / $a" | bc -l)
  echo "P(X ≥ $a) ≤ $result"
}

# Function to display options and prompt the user for input
choose_law() {
  echo "Please choose a law to calculate or learn about:"
  echo "1. Addition Law of Probability"
  echo "2. Multiplication Law of Probability"
  echo "3. Complement Law"
  echo "4. Conditional Probability"
  echo "5. Bayes' Theorem"
  echo "6. Law of Total Probability"
  echo "7. Weak Law of Large Numbers"
  echo "8. Central Limit Theorem"
  echo "9. Markov's Inequality"
  echo "10. Exit"
  read -p "Enter the number of your choice: " choice

  case $choice in
    1)
      addition_law
      ;;
    2)
      multiplication_law
      ;;
    3)
      complement_law
      ;;
    4)
      conditional_probability
      ;;
    5)
      bayes_theorem
      ;;
    6)
      total_probability
      ;;
    7)
      weak_law_large_numbers
      ;;
    8)
      central_limit_theorem
      ;;
    9)
      markov_inequality
      ;;
    10)
      echo "Exiting..."
      exit 0
      ;;
    *)
      echo "Invalid choice, please try again."
      choose_law
      ;;
  esac
}

# Main loop
while true; do
  choose_law
done
