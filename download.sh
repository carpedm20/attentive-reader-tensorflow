#!/bin/bash
if [ ! -d ./data ] 
then
  echo "Create data directory..."
  mkdir -p ./data
fi

echo "Download cnn.tgz..."
curl 'https://doc-0g-74-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/i3sb5n8scpjl555une0dg85kkddclv17/1453154400000/00708016728989131340/*/0BwmD_VLjROrfTTljRDVZMFJnVWM?e=download' -H 'pragma: no-cache' -H 'accept-encoding: gzip, deflate, sdch' -H 'accept-language: ko-KR,ko;q=0.8,en-US;q=0.6,en;q=0.4' -H 'upgrade-insecure-requests: 1' -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.111 Safari/537.36' -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'cache-control: no-cache' -H 'authority: doc-0g-74-docs.googleusercontent.com' -H 'referer: https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTTljRDVZMFJnVWM' --compressed -o cnn.tgz
echo "Unzip cnn.tgz..."
tar -xzvf cnn.tgz -C data/

echo "Download dailymail.tgz"
curl 'https://doc-04-74-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/52d917egbllm5ndafjnhbla5jdpodf0r/1453154400000/00708016728989131340/*/0BwmD_VLjROrfN0xhTDVteGQ3eG8?e=download' -H 'pragma: no-cache' -H 'accept-encoding: gzip, deflate, sdch' -H 'accept-language: ko-KR,ko;q=0.8,en-US;q=0.6,en;q=0.4' -H 'upgrade-insecure-requests: 1' -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.111 Safari/537.36' -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'cache-control: no-cache' -H 'authority: doc-04-74-docs.googleusercontent.com' -H 'referer: https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfN0xhTDVteGQ3eG8' --compressed -o dailymail.tgz
echo "Unzip cnn.tgz..."
tar -xzvf dailymail.tgz -C data/
