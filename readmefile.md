**Start Here:**
In this project we will start with making directory in Ubuntu Linux Distribution. My Distro is currently installed in my virtual box and windows as the host.


**installing Git in Ubuntu Distro:** 

**- [ sudo apt-get install Git]** 

To make our journey easier, I will connect my Github to my local Git to make the interaction easier. However, before I connect my Git and Github, 
I will start with the installation Process.

![Image](https://github.com/user-attachments/assets/b073847e-732c-41bb-8233-95c7c8caddea)

**- [ git --version]** 

To ensure we have installed the application correctly, we are going to ensure the appropriate version exists.

![Image](https://github.com/user-attachments/assets/8afae433-f7c3-452d-810b-4de46d0d850b)

****- [ git --help]** or  [ git --help rm [what you want to learn about. ]** 

If you are curious individual like me, I am sure you're curious what other commands are available in Git command parameters. in order to find list of commands and parameters or elements available feel free to do the following.

![Image](https://github.com/user-attachments/assets/3cd32c78-1400-455f-b0b5-091676545c2b)

**Creating Target Folder:**
To make your development journey easier, I find it rewarding to start with a development folder or environment for each project. Later on we'll explore things such as **Python venv OR virtualenv** which will make it easier for us to create virtual environments that will enable us to install targeted package for our projects without impacting the core machine.

 **-[sudo mkdir Github_Projects]**  
 
 This command will assist us to build a folder within our current directory. 
 
![Image](https://github.com/user-attachments/assets/0bc7826c-a400-4ee0-a583-053d58f5a29e)

Please note we are adding **sudo** because we are not at root. To avoid this we can every-time we run a command, we can give the current user ownership of current directory - [ ] **sudo chown userinfo directoryname** or if you trust the user to root they could enter and once done they can type simply**sudo -s**  in the terminal.

![Image](https://github.com/user-attachments/assets/4fcfe472-3549-464d-ad4d-38fc1f4b077a)

To determine which user you're, you can utilize the whoami command.

![Image](https://github.com/user-attachments/assets/ff475e68-e2d6-4558-baf4-ad579d6950b5)


- [ cd into_directory] Now we created the directory, we can switch into the folder. To see what is the folder, we can utilize - [ ls] to list any sub-folders or files inside the newly created directory.   

![Image](https://github.com/user-attachments/assets/93a4fd17-2efc-4412-b24f-60d2500350ae)


**Connecting Git to Github:**

Before we start connecting the local and external technologies, let us configure the username for the git directory. Remember, we are doing this because different users from the organization or collaborators could contribute to the project and this will allow us to determine who is doing the different commits. git config --global user.name "Your Desired Name" 

![Image](https://github.com/user-attachments/assets/f85ad21e-ab5d-402d-8b4f-550d5575b0e3)

The above command can even execute the email and any other information that you want to add. instead of using the master repository, we can also change the default repository with the following commands:

![Image](https://github.com/user-attachments/assets/3e37bfe5-e1a8-416e-99bb-8861726e2efb)

Again be curious and as true for any package, library, or software utilize its manual or --help functions. don't stop exploring!

Note: Remember, we could have created the repository from the Github website first manually.

Now, we set up the directory, configured our environment, and changed some default branches. Let's start pushing our branch to Github. Because we will be using this quite often, I am going to create a script that we can call as needed. off course, we will need to make change depending on what we need but let start somewhere.

Initialization Local Repository  / Commit  
  


