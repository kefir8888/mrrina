#!/bin/sh
#!/usr/bin/expect
#!/usr/bin/spawn
#!/usr/bin/send
spawn ssh nao@192.168.1.29
scp nao@192.168.1.29:/home/nao/recordings/cameras/kek.jpg .
send "nao"