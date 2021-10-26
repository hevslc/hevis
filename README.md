# HeVis





### Quickstart

* Ensure you have [Python 3.6+](https://www.python.org/downloads/), [Node.js](https://nodejs.org), and [npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) installed.
* Clone this repo.
* Create a new Python virtual environment for the template:
```
$ cd hevis
$ python -m venv venv  # create venv
$ # activate venv:
$ . venv/bin/activate # OR
$ cd venv/Scripts & activate   
$ pip install -r requiriments.txt # install dependences
```
* Back to hevis directory, initialize and run the component template frontend:
```
$ cd frontend
$ npm install    # Install npm dependencies
$ npm run start  # Start the Webpack dev server
```
* If necessary, run the following command
```
$ export NODE_OPTIONS="--max-old-space-size=8192"
```
* From a separate terminal, run the HeVis Streamlit app:
```
$ cd hevis
$ . venv/bin/activate # OR
$ cd venv/Scripts & activate  
$ streamlit run hevis.py  # run the example server
```

* Observation: Downloading .eps files requires installing poopler
