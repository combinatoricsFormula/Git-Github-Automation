#!/bin/bash

# Function to check if a command exists and install it if not
install_if_missing() {
    if ! command -v "$1" &> /dev/null; then
        echo "$1 is not installed. Installing..."
        sudo apt update && sudo apt install -y "$1" || {
            echo "Failed to install $1. Please ensure you have sudo privileges.";
            exit 1;
        }
    else
        echo "$1 is already installed."
    fi
}

# Function to generate SSH key if not exists
generate_ssh_key() {
    local email="$1"
    # Check if SSH keys already exist
    if [ ! -f ~/.ssh/id_rsa ] || [ ! -f ~/.ssh/id_rsa.pub ]; then
        echo "SSH key not found. Generating a new SSH key..."
        ssh-keygen -t rsa -b 4096 -C "$email" -f ~/.ssh/id_rsa -N "" || {
            echo "Failed to generate SSH key. Please check your SSH configuration.";
            exit 1;
        }
    else
        echo "SSH key already exists. Skipping SSH key generation."
    fi

    # Ensure the correct permissions for the SSH key files
    chmod 600 ~/.ssh/id_rsa
    chmod 644 ~/.ssh/id_rsa.pub

    # Display the public SSH key and prompt the user to add it to GitHub
    echo "Your public SSH key is:"
    cat ~/.ssh/id_rsa.pub
    echo "Please add this SSH key to your GitHub account under 'Settings' -> 'SSH and GPG keys'."
    
    # Start SSH agent and add the key
    echo "Starting the SSH agent..."
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_rsa || {
        echo "Failed to add SSH key to the agent. Please check your SSH configuration.";
        exit 1;
    }

    # Verify SSH connection to GitHub
    verify_ssh_connection
}

# Function to verify SSH connection to GitHub
verify_ssh_connection() {
    echo "Verifying SSH connection to GitHub..."
    ssh -T git@github.com
    if [ $? -ne 1 ]; then
        echo "SSH connection to GitHub failed."
        read -p "Do you want to retry the SSH connection? (y/n): " retry_choice
        if [[ "$retry_choice" == "y" || "$retry_choice" == "Y" ]]; then
            verify_ssh_connection
        else
            echo "Proceeding without SSH verification."
        fi
    else
        echo "SSH connection to GitHub successful!"
    fi
}

# Function to generate GPG key if not exists
generate_gpg_key() {
    # Check if GPG keys already exist
    if [ ! -f ~/.gnupg/pubring.kbx ]; then
        echo "No GPG key found. Generating a new GPG key..."
        gpg --full-generate-key || {
            echo "Failed to generate GPG key. Please ensure GPG is installed.";
            exit 1;
        }
        GPG_KEY=$(gpg --list-secret-keys --keyid-format LONG | grep -oP "(?<=/)[A-F0-9]{16}" | head -n 1)
        git config --global user.signingkey "$GPG_KEY"
        echo "GPG key generated and configured for Git."
    else
        echo "GPG key already exists. Skipping GPG key generation."
    fi
    echo "Your GPG key ID is:"
    gpg --list-secret-keys --keyid-format LONG

    # Set GPG for commits
    git config --global commit.gpgSign true
}

# Function to create or select a GitHub repository
create_or_select_github_repo() {
    local username="$1"
    local token="$2"
    local repo_name="$3"
    
    # Fetch the list of repositories for the user
    repos=$(curl -s -H "Authorization: token $token" https://api.github.com/user/repos)
    repo_exists=$(echo "$repos" | grep -q "\"name\": \"$repo_name\"" && echo "true" || echo "false")

    if [ "$repo_exists" == "true" ]; then
        echo "Repository '$repo_name' already exists on GitHub."
    else
        echo "Repository '$repo_name' does not exist. Creating a new one..."
        response=$(curl -s -X POST -H "Authorization: token $token" -d "{\"name\":\"$repo_name\"}" https://api.github.com/user/repos)
        if echo "$response" | grep -q '"full_name":'; then
            echo "Repository '$repo_name' created successfully on GitHub."
        else
            echo "Failed to create repository. Response from GitHub: $response"
            exit 1
        fi
    fi

    # List all branches of the selected repository
    echo "Fetching branches for repository '$repo_name'..."
    branches=$(curl -s -H "Authorization: token $token" "https://api.github.com/repos/$username/$repo_name/branches")
    branch_names=$(echo "$branches" | jq -r '.[].name')

    echo "Existing branches in '$repo_name':"
    echo "$branch_names"

    # Ask user to select a branch or create a new one
    echo "Enter the branch you want to push to (or create a new branch):"
    read branch_name
    branch_name=${branch_name:-main}  # Default to 'main' if nothing is entered

    # Check if the branch exists on GitHub, if not create it
    if ! echo "$branch_names" | grep -q "$branch_name"; then
        echo "Branch '$branch_name' does not exist on GitHub. Creating the branch..."
        git checkout -b "$branch_name"
        git push -u origin "$branch_name"
        echo "Branch '$branch_name' created and pushed to GitHub."
    else
        echo "Branch '$branch_name' exists. Switching to it and pushing changes..."
        git checkout "$branch_name"
    fi

    # Set the remote repository URL and push changes
    git remote add origin "git@github.com:$username/$repo_name.git" || git remote set-url origin "git@github.com:$username/$repo_name.git"
    git push -u origin "$branch_name" || {
        echo "Failed to push changes to GitHub.";
        exit 1;
    }

    # Verify remote repository
    echo "Verifying remote repository setup..."
    git remote -v
    git remote show origin
}

# Main script starts here
echo "Starting the setup process..."

# Step 1: Install Dependencies
DEPENDENCIES=("git" "curl" "ssh-keygen" "gpg" "jq")
for dep in "${DEPENDENCIES[@]}"; do
    install_if_missing "$dep"
done

# Step 2: Check Credentials
echo "Checking credentials..."
echo "Enter your email address for SSH key generation:"
read user_email
generate_ssh_key "$user_email"
generate_gpg_key

# Step 3: Ask User for Repository Setup
echo "Do you want to use the current directory as the repository, create another directory, or choose from your GitHub repositories? (current/new/github)"
read repo_choice

if [ "$repo_choice" == "current" ]; then
    echo "Using the current directory as the repository."
    echo "Current directory: $(pwd)"
    echo "Files in the current directory:"
    ls -l
    echo "Enter the file name to modify (or provide a new file name with extension):"
    read file_name
    nano "$file_name"
elif [ "$repo_choice" == "new" ]; then
    echo "Enter the name of the new repository directory:"
    read new_repo_name
    mkdir -p "$new_repo_name"
    cd "$new_repo_name" || exit 1
    echo "Enter the file name to create (with extension):"
    read new_file_name
    nano "$new_file_name"
elif [ "$repo_choice" == "github" ]; then
    echo "Enter your GitHub username:"
    read github_username
    echo "Enter your GitHub Personal Access Token (PAT):"
    read -s github_token
    echo "Fetching your GitHub repositories..."
    repos=$(curl -s -H "Authorization: token $github_token" https://api.github.com/user/repos | jq -r '.[].name')
    echo "Your GitHub repositories:"
    echo "$repos"
    echo "Enter the name of the repository to use:"
    read github_repo_name
    create_or_select_github_repo "$github_username" "$github_token" "$github_repo_name"
    echo "Fetching branches for repository '$github_repo_name'..."
    branches=$(curl -s -H "Authorization: token $github_token" "https://api.github.com/repos/$github_username/$github_repo_name/branches" | jq -r '.[].name')
    echo "Existing branches in '$github_repo_name':"
    echo "$branches"
    echo "Enter the branch you want to use (or create a new branch):"
    read branch_name
    branch_name=${branch_name:-main}  # Default to 'main' if nothing is entered
    if ! echo "$branches" | grep -q "$branch_name"; then
        echo "Branch '$branch_name' does not exist. Creating it..."
        git checkout -b "$branch_name"
        git push -u origin "$branch_name"
    else
        git checkout "$branch_name"
    fi
    echo "Enter the file name to modify (or provide a new file name with extension):"
    read file_name
    nano "$file_name"
else
    echo "Invalid choice. Exiting."
    exit 1
fi

# Step 4: Commit Changes
echo "Do you want to commit changes from the entire directory or a specific file? (all/specific)"
read commit_choice
if [ "$commit_choice" == "all" ]; then
    git add .
else
    echo "Enter the file name to commit:"
    read commit_file
    git add "$commit_file"
fi
echo "Enter your commit message:"
read commit_message
git commit -S -m "$commit_message" || { echo "Failed to sign commit with GPG."; exit 1; }

# Step 5: Push Changes to GitHub
echo "Pushing changes to GitHub..."
git push origin "$branch_name" || { echo "Failed to push changes to GitHub."; exit 1; }

echo "Process complete."

