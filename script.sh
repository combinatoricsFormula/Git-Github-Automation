#!/bin/bash
# 1 Installing Git
	echo "starting the Git step up process..."
	###let start with installing for Ubuntu which is 
	###currently running on my windows VBOX host machine
	echo "checking for previous installs..."
	if ! command -v git &> /dev/null; then
		echo "Git not installed"
		sudo apt update
		sudo apt isntall -y git

	else
		echo "Git is installed"

	fi
	### Let us make sure istallation is successful
	git --version


# 2 Configuration Process

	echo "Configuration Git User Name and Email..."
	git config --global user.name  "Amear Hussein Mathews"
	git config --global user.email "amear.h.mathews@gmail.com"
	git config --global credential.helper cache
	git config --global color.ui auto
	git config --global commit.gpgSign true


# 3 My Favorite Editor - Choose yours wisely
	git config --global core.editor "nano"


# 4 Let us create Project Directory
	echo "What do you want to call this chapter? "
	read chapter
	if [ -z "$chapter" ]; then
    		echo "test directory created. You can change name later"
    		mkdir test
	else
    		echo "$chapter has been created"
    		mkdir "$chapter"
	fi

# 5 Intitialization / staging / commit phases

	git init
	echo "Do you want to add or commit all files?"
	echo "For all files, Enter: '.' otherwise provide a file name."
	read commitall
	if [[ "$commitall" == "." ]]; then
    		git add .
	else
    		git add "$commitall"
	fi



# 6 SSH Process

	if [ ! -f ~/.ssh/id_rsa ]; then
		echo "No SSH Key Found. Generate a new SSH Key..."
		ssh-keygen -t rsa -b 4096 -C "amear.h.mathews@gmail.com" -f ~/.ssh/id_rsa -N ""
	else
		echo " It exist.SKipping..."

	fi

	echo " Your public keys are: "
	cat ~/.ssh/id_rsa.pub
	echo " Now add to your github under the ssh key seetings"

# 7 Installation GPG

	# Check if a GPG key exists by looking for the directory where GPG keys are stored
	if [ ! -f ~/.gnupg/pubring.kbx ]; then
    		echo "No GPG key found. Generating a new key..."
    		gpg --full-generate-key  # Prompt user to create a GPG key
    		GPG_KEY=$(gpg --list-secret-keys --keyid-format LONG | grep -oP "(?<=/)[A-F0-9]{16}" | head -n 1)  # Get the generated key ID
    		git config --global user.signingkey "$GPG_KEY"  # Set the GPG key for signing
    		echo "GPG key generated and configured for Git."
	else
    		echo "GPG key already exists. Skipping key generation."
		
	fi
	echo " Your public keys are: "
	cat ~/.gnupg/pubring.kbx
	echo " Now add to your github under the ssh key seetings"


# 8 Status Check

	echo "Your commit description"
	read comments
	git commit -m "$comments"
	git branch -M main
	git status

	url ="git@github.com:Combinatorics-AMEARMathews/git_course.git"
	echo "Connecting to the remote Github repository : $url"
	git remote add Git_Course $url

# 9 pushing changes to Github

	echo " Pushing to github"
	git branch -a
	git push -u git_course main

# 10 Verifying Github

	git remote -v

	git remote show git_course
