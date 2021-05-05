# Made Production ML

Install
~~~
git clone https://github.com/made-ml-in-prod-2021/nickdndev.git
cd online_inference/

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

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

python -m src.make_request --host "localhost" --port "8000" --dataset_path "data/heart.csv"

or use default parameters

python -m src.make_request
~~~

Test:
~~~
docker run -p 8000:8000 nickml/online_inference:v1

python -m src.make_request --host "localhost" --port "8000" --dataset_path "data/heart.csv"

or use default parameters

python -m src.make_request
~~~

