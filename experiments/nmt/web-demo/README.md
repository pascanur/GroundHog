Neural Machine Translation (Web-based Demostration System)
-------------------------- -------------------------------

This folder contains scripts that can be used to demonstrate the trained model
via web interface.

####Code Structure

- index.html contains the front-end interface
- sample_server.py is an HTTP server version of ../sample.py
- sampler.php queries sample_server.py to get a translation
- tokenizer.perl tokenizes a sentence
- detokenizer.perl de-tokenizes a sentence
- both perl files use the prefix files in nonbreaking_prefixes/

This works with bootstrap v3.2.0 (http://getbootstrap.com/).
  
####How to run a demo server

On your computing server, run
```
./sample_server.py --port=8888 --beam-search --state=your_state_file.pkl your_model_file.npz
```
This will start a sampling server listening on 0.0.0.0:8888.

On your front-end web server, put index.html and sampler.php. Inside
sampler.php, fix the following line to point to your computing server:
```
$url = 'http://your_computing_server:8888/?source='.urlencode($source);
```

