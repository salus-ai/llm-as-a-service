# Install kubectl
echo "Installing kubectl..."
# Download kubectl binary
curl -O https://s3.us-west-2.amazonaws.com/amazon-eks/1.27.1/2023-04-19/bin/linux/amd64/kubectl

# Download the checksum for verification
curl -O https://s3.us-west-2.amazonaws.com/amazon-eks/1.27.1/2023-04-19/bin/linux/amd64/kubectl.sha256

# Verify the checksum
echo "Verifying the checksum..."
sha256sum -c kubectl.sha256
openssl sha1 -sha256 kubectl

# Make the binary executable
chmod +x ./kubectl

# Move the binary to user binary directory and update PATH
mkdir -p $HOME/bin && cp ./kubectl $HOME/bin/kubectl && export PATH=$HOME/bin:$PATH

# Update PATH in bashrc so it persists across sessions
echo 'export PATH=$HOME/bin:$PATH' >> ~/.bashrc

# Check the version to verify successful installation
kubectl version --short --client
echo "kubectl installed successfully."

# Install eksctl
echo "Installing eksctl..."
ARCH=amd64
PLATFORM=$(uname -s)_$ARCH

# Download eksctl binary
curl -sLO "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$PLATFORM.tar.gz"

# (Optional) Verify checksum
echo "Verifying the checksum..."
curl -sL "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_checksums.txt" | grep $PLATFORM | sha256sum --check

# Extract the binary and move it to /usr/local/bin
tar -xzf eksctl_$PLATFORM.tar.gz -C /tmp && rm eksctl_$PLATFORM.tar.gz
sudo mv /tmp/eksctl /usr/local/bin
echo "eksctl installed successfully."

# Install Docker
echo "Installing Docker..."
sudo yum update
sudo yum install docker

# Add ec2-user to docker group so the user can execute docker commands
sudo usermod -a -G docker ec2-user
id ec2-user
newgrp docker

# Enable and start Docker service
sudo systemctl enable docker.service
sudo systemctl start docker.service
echo "Docker installed successfully."

# Install Anaconda
# echo "Installing Anaconda..."
# wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
# bash Anaconda3-2023.03-1-Linux-x86_64.sh
# echo "Anaconda installed successfully."

# Install Paradigm 
echo "Installing Paradigm..."
git clone https://github.com/ParadigmAI/paradigm.git
cd paradigm
chmod +x install-aws.sh
./install-aws.sh
paradigm --help
echo "Paradigm installed successfully."


