<?php

$source=$_GET["source"];
$url = 'http://eos12:8888/?source='.urlencode($source);
$out = file_get_contents($url);

echo urldecode($out);

?>

