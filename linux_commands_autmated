#!/bin/bash

# Define a list of commands and their descriptions
declare -A commands
commands=(
    ["alias"]="Create an alias for a command"
    ["at"]="Schedule a command to run at a later time"
    ["awk"]="Pattern scanning and processing language"
    ["basename"]="Strip directory and suffix from filenames"
    ["cal"]="Display a calendar"
    ["cat"]="Concatenate and display files"
    ["cd"]="Change the current directory"
    ["chmod"]="Change file modes or Access Control Lists"
    ["chown"]="Change file owner and group"
    ["cp"]="Copy files and directories"
    ["crontab"]="Schedule periodic background jobs"
    ["curl"]="Transfer data from or to a server"
    ["cut"]="Remove sections from each line of files"
    ["df"]="Report file system disk space usage"
    ["diff"]="Compare files line by line"
    ["du"]="Estimate file space usage"
    ["echo"]="Display a line of text"
    ["find"]="Search for files in a directory hierarchy"
    ["grep"]="Print lines that match patterns"
    ["head"]="Output the first part of files"
    ["kill"]="Send a signal to a process"
    ["less"]="View file contents one screen at a time"
    ["ln"]="Create hard and symbolic links"
    ["ls"]="List directory contents"
    ["man"]="Display the manual for a command"
    ["mkdir"]="Create directories"
    ["mv"]="Move or rename files and directories"
    ["ps"]="Report a snapshot of current processes"
    ["pwd"]="Print the name of the current working directory"
    ["rm"]="Remove files or directories"
    ["rmdir"]="Remove empty directories"
    ["scp"]="Secure copy (remote file copy program)"
    ["sed"]="Stream editor for filtering and transforming text"
    ["ssh"]="OpenSSH remote login client"
    ["tail"]="Output the last part of files"
    ["tar"]="Archive files"
    ["touch"]="Change file timestamps"
    ["uname"]="Print system information"
    ["uptime"]="Tell how long the system has been running"
    ["wget"]="Non-interactive network downloader"
    ["who"]="Show who is logged on"
    ["xargs"]="Build and execute command lines from standard input"
)

# Function to search for a command by keyword
search_command() {
    local keyword=$1
    for cmd in "${!commands[@]}"; do
        if [[ "${commands[$cmd]}" == *"$keyword"* ]]; then
            echo $cmd
            return
        fi
    done
    echo "No command found for keyword: $keyword"
}

# Prompt the user for action
echo "What do you intend to do?"
read action

# Search for the command based on the action keyword
command=$(search_command "$action")

if [[ $command == "No command found for keyword: $action" ]]; then
    echo $command
    exit 1
fi

# Confirm the command with the user
echo "Do you want to execute the command '$command' for the action '$action'? (yes/no)"
read confirmation

# Execute the command if confirmed
if [[ $confirmation == "yes" ]]; then
    echo "Executing: $command"
    $command
else
    echo "Command not executed."
fi
