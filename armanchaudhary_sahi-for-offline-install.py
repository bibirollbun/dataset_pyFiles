!pip download -d ./packages sahi
!tar cfvz archive.tar.gz ./packages


!tar xfvz archive.tar.gz
!pip install --no-index --find-links=./packages sahi
!rm -rf ./packages 

