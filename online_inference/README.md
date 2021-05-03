# Made Production ML

Build docker:
~~~
docker build -t nickml/online_inference:v1 .
~~~

Pull docker
~~~
docker pull nickml/online_inference:v1
~~~

Run docker and make requets  
~~~
docker run -p 8000:8000 nickml/online_inference:v1
python make_request.py 
~~~
