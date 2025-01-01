#!/bin/bash

# Function to check if a command exists and install it if missing
install_if_missing() {
    if ! command -v "$1" &> /dev/null; then
        echo "$1 is not installed. Installing..."
        sudo apt update && sudo apt install -y "$1" || {
            echo "Failed to install $1. Please ensure you have sudo privileges."
            exit 1
        }
    else
        echo "$1 is already installed."
    fi
}

# Function to generate SSH key if not exists
generate_ssh_key() {
    local email="$1"
    if [ ! -f ~/.ssh/id_rsa ] || [ ! -f ~/.ssh/id_rsa.pub ]; then
        echo "SSH key not found. Generating a new SSH key..."
        ssh-keygen -t rsa -b 4096 -C "$email" -f ~/.ssh/id_rsa -N "" || {
            echo "Failed to generate SSH key. Please check your SSH configuration."
            exit 1
        }
    else
        echo "SSH key already exists. Skipping SSH key generation."
    fi

    chmod 600 ~/.ssh/id_rsa
    chmod 644 ~/.ssh/id_rsa.pub

    echo "Your public SSH key is:"
    cat ~/.ssh/id_rsa.pub
    echo "Please add this SSH key to your GitHub account under 'Settings' -> 'SSH and GPG keys'."
    
    echo "Starting the SSH agent..."
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_rsa || {
        echo "Failed to add SSH key to the agent. Please check your SSH configuration."
        exit 1
    }

    # Upload SSH key to GitHub
    upload_ssh_key_to_github
    verify_ssh_connection
}

# Function to upload SSH public key to GitHub
upload_ssh_key_to_github() {
    echo "Enter your GitHub Personal Access Token (PAT) to upload the SSH key:"
    read -s github_token

    # Option to save the PAT to a file for later use
    echo "Would you like to save your GitHub Personal Access Token (PAT) for future use? (y/n)"
    read save_pat_choice
    if [[ "$save_pat_choice" == "y" || "$save_pat_choice" == "Y" ]]; then
        # Save the token to a file (ensure the file is excluded from git commits)
        echo "$github_token" > ~/.github_pat.txt
        echo "GitHub Personal Access Token saved to ~/.github_pat.txt for future use."
    fi

    ssh_key_title="My SSH Key"
    ssh_key=$(cat ~/.ssh/id_rsa.pub)

    # GitHub API endpoint to upload SSH key
    curl -X POST -H "Authorization: token $github_token" \
        -d "{\"title\":\"$ssh_key_title\", \"key\":\"$ssh_key\"}" \
        "https://api.github.com/user/keys" || {
            echo "Failed to upload SSH key to GitHub. Please check your token and GitHub settings."
            exit 1
        }

    echo "SSH public key uploaded to GitHub successfully!"
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

# Function to retrieve PAT from file if it exists
retrieve_github_pat() {
    if [ -f ~/.github_pat.txt ]; then
        echo "Retrieving saved GitHub Personal Access Token..."
        github_token=$(cat ~/.github_pat.txt)
        echo "GitHub Personal Access Token retrieved successfully!"
    else
        echo "No saved GitHub Personal Access Token found."
        return 1
    fi
}

# Function to generate GPG key if not exists
generate_gpg_key() {
    if [ ! -f ~/.gnupg/pubring.kbx ]; then
        echo "No GPG key found. Generating a new GPG key..."
        gpg --full-generate-key || {
            echo "Failed to generate GPG key. Please ensure GPG is installed."
            exit 1
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

# Function to allow the user to manually enter their personal GPG key for commit signing
set_gpg_key() {
    echo "Enter your GPG key ID for commit signing (Leave blank to use the default key):"
    read user_gpg_key
    if [[ -n "$user_gpg_key" ]]; then
        git config --global user.signingkey "$user_gpg_key"
        echo "Using the GPG key ID $user_gpg_key for commit signing."
    else
        echo "Using the default GPG key for commit signing."
    fi
}

# Function to configure GitHub user settings
configure_github_user() {
    echo "Enter your GitHub username:"
    read github_username
    git config --global user.name "$github_username"
    
    echo "Enter your GitHub email address:"
    read github_email
    git config --global user.email "$github_email"

    echo "GitHub user configuration complete."
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
    echo "Enter the branch you want to push to (or create a new branch, default is 'main'):" 
    read branch_name
    branch_name=${branch_name:-main}  # Default to 'main' if nothing is entered

    if echo "$branch_names" | grep -q "$branch_name"; then
        echo "Branch '$branch_name' already exists. You can continue with it or create a new one. Would you like to create a new branch? (y/n)"
        read create_new_branch
        if [[ "$create_new_branch" == "y" || "$create_new_branch" == "Y" ]]; then
            echo "Creating new branch '$branch_name'..."
            git checkout -b "$branch_name"
            git push -u origin "$branch_name"
            echo "Branch '$branch_name' created and pushed to GitHub."
        else
            git checkout "$branch_name"
            echo "Using existing branch '$branch_name'."
        fi
    else
        # Branch doesn't exist, so create it
        echo "Branch '$branch_name' does not exist. Creating the branch..."
        git checkout -b "$branch_name"
        git push -u origin "$branch_name"
        echo "Branch '$branch_name' created and pushed to GitHub."
    fi

    # Set the remote repository URL and push changes
    git remote add origin "git@github.com:$username/$repo_name.git" || git remote set-url origin "git@github.com:$username/$repo_name.git"
    push_output=$(git push -u origin "$branch_name" 2>&1)
    push_status=$?

    if [ $push_status -ne 0 ]; then
        echo "Push failed with the following error:"
        echo "$push_output"
        if echo "$push_output" | grep -q "Authentication failed"; then
            echo "Authentication failed. Please check your SSH key or GitHub token."
        elif echo "$push_output" | grep -q "non-fast-forward"; then
            echo "Push failed: The local branch is behind the remote branch. Try pulling the changes before pushing."
        elif echo "$push_output" | grep -q "rejected"; then
            echo "Push failed: The push was rejected by the remote repository. Verify your permissions and branch configuration."
        else
            echo "An unknown error occurred during push. Please review the above message for more details."
        fi
        exit 1
    else
        echo "Push successful!"
    fi

    # Verify remote repository
    echo "Verifying remote repository setup..."
    git remote -v
    git remote show origin
}

# Function to show the file that was modified
show_file_modifications() {
    local file_name="$1"
    echo "File '$file_name' was modified."
    echo "Changes made:"
    git diff -- "$file_name"
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
generate_ssh_key "your_email@example.com"
generate_gpg_key
set_gpg_key
configure_github_user

# Step 3: Ask User for Repository Setup
echo "Do you want to use the current directory as the repository, create another directory, or choose from your GitHub repositories? (current/new/github)"
read repo_choice

if [ "$repo_choice" == "current" ]; then
    echo "Using the current directory as the repository."
    git init
    echo "Enter the file name to modify (or provide a new file name with extension):"
    read file_name
    nano "$file_name"
    show_file_modifications "$file_name"
    git checkout -b main
    git push -u origin main

elif [ "$repo_choice" == "new" ]; then
    echo "Enter the name of the new repository directory:"
    read new_repo_name
    mkdir -p "$new_repo_name"
    cd "$new_repo_name" || exit 1
    git init
    echo "Enter the file name to create (with extension):"
    read new_file_name
    nano "$new_file_name"
    git checkout -b main
    git push -u origin main

elif [ "$repo_choice" == "github" ]; then
    # Try to retrieve the PAT from file if exists
    retrieve_github_pat
    if [ $? -ne 0 ]; then
        echo "Enter your GitHub Personal Access Token (PAT):"
        read -s github_token
    fi
    echo "Fetching your GitHub repositories..."
    repos=$(curl -s -H "Authorization: token $github_token" https://api.github.com/user/repos | jq -r '.[].name')
    echo "Your GitHub repositories:"
    echo "$repos"
    echo "Enter the name of the repository to use:"
    read github_repo_name
    create_or_select_github_repo "$github_username" "$github_token" "$github_repo_name"
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
