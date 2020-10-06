//////////////////////init///////////////////////////////
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"  

  git config --global user.email "guimingthu@163.com"
  git config --global user.name "guiming"

git init
git remote add origin git@github.com:guiming-shi/assignment-algorithm.git
git pull origin master

//////////////////////everytime///////////////////////////////
git remote add origin git@github.com:guiming-shi/assignment-algorithm.git
git pull origin master
git add .
git commit -m "first commit"
git branch -M master
git remote add origin git@github.com:guiming-shi/assignment-algorithm.git
git push -u origin master
                



