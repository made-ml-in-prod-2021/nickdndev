~~~
docker build -t nickml/online_inference:v1 .
docker run -p 9000:9000 nickml/online_inference:v1
python make_request.py 

~~~
