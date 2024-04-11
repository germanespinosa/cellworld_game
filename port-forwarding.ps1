If (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
  $arguments = "& '" + $myinvocation.mycommand.definition + "'"
  Start-Process powershell -Verb runAs -ArgumentList $arguments
  Break
}
$ifconfig = wsl.exe ifconfig eth0
$remoteport = $ifconfig| Select-String -Pattern "inet "
$found = $remoteport -match '\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}';

if ($found) {
  $remoteport = $matches[0];
}
else {
  Write-Output "IP address could not be found";
  exit;
}




Invoke-Expression "netsh interface portproxy reset";

$portl = $args
$portl+= "6006"

for ($i = 0; $i -lt $portl.length; $i++) {
  $port = $portl[$i];
  Invoke-Expression "netsh interface portproxy add v4tov4 listenport=$port connectport=$port connectaddress=$remoteport";
  Invoke-Expression "netsh advfirewall firewall add rule name=$port dir=in action=allow protocol=TCP localport=$port";
}

Invoke-Expression "netsh interface portproxy show v4tov4";