import socket
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor
import netifaces
import ipaddress
from urllib.parse import urlparse

class LGTVScanner:
    def __init__(self):
        self.found_tvs = []
        self.timeout = 2
        
    def get_local_network(self):
        """Get the local network range"""
        try:
            # Get default gateway interface
            gateways = netifaces.gateways()
            default_gateway = gateways['default'][netifaces.AF_INET]
            interface = default_gateway[1]
            
            # Get network info for the interface
            addrs = netifaces.ifaddresses(interface)
            ipv4_info = addrs[netifaces.AF_INET][0]
            
            ip = ipv4_info['addr']
            netmask = ipv4_info['netmask']
            
            # Calculate network range
            network = ipaddress.IPv4Network(f"{ip}/{netmask}", strict=False)
            return str(network)
            
        except Exception as e:
            print(f"Error getting network range: {e}")
            return "192.168.1.0/24"  # Default fallback
    
    def check_lg_tv_port(self, ip, port=3001):
        """Check if LG TV WebOS port is open"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex((ip, port))
            sock.close()
            return result == 0
        except:
            return False
    
    def check_lg_tv_http(self, ip):
        """Check for LG TV via HTTP requests"""
        try:
            # Try common LG TV HTTP endpoints
            urls = [
                f"http://{ip}:3000/udap/api/data?target=netrcu.model",
                f"http://{ip}:3001/",
                f"http://{ip}:8080/"
            ]
            
            for url in urls:
                try:
                    response = requests.get(url, timeout=self.timeout)
                    content = response.text.lower()
                    headers = str(response.headers).lower()
                    
                    # Look for LG-specific indicators
                    lg_indicators = ['lg', 'webos', 'netcast', 'smarttv']
                    if any(indicator in content or indicator in headers for indicator in lg_indicators):
                        return True, url
                except:
                    continue
                    
            return False, None
        except:
            return False, None
    
    def get_device_info(self, ip):
        """Try to get device information"""
        try:
            # Try to get hostname
            hostname = socket.gethostbyaddr(ip)[0]
        except:
            hostname = "Unknown"
        
        # Try to get MAC address (requires root on some systems)
        mac_address = "Unknown"
        try:
            import subprocess
            result = subprocess.run(['arp', '-n', ip], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if ip in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            mac_address = parts[2]
                        break
        except:
            pass
        
        return hostname, mac_address
    
    def scan_ip(self, ip):
        """Scan a single IP address for LG TV"""
        ip_str = str(ip)
        
        # Check if WebOS port is open
        if self.check_lg_tv_port(ip_str):
            # Get additional info
            hostname, mac_address = self.get_device_info(ip_str)
            is_lg_http, detected_url = self.check_lg_tv_http(ip_str)
            
            tv_info = {
                'ip': ip_str,
                'hostname': hostname,
                'mac_address': mac_address,
                'port_3001_open': True,
                'http_detection': is_lg_http,
                'detected_url': detected_url
            }
            
            self.found_tvs.append(tv_info)
            print(f"[+] Potential LG TV found: {ip_str}")
            if hostname != "Unknown":
                print(f"    Hostname: {hostname}")
            if is_lg_http:
                print(f"    HTTP Detection: Success ({detected_url})")
        
        # Also check HTTP even if port isn't open (some might use different ports)
        elif not self.check_lg_tv_port(ip_str):
            is_lg_http, detected_url = self.check_lg_tv_http(ip_str)
            if is_lg_http:
                hostname, mac_address = self.get_device_info(ip_str)
                tv_info = {
                    'ip': ip_str,
                    'hostname': hostname,
                    'mac_address': mac_address,
                    'port_3001_open': False,
                    'http_detection': True,
                    'detected_url': detected_url
                }
                self.found_tvs.append(tv_info)
                print(f"[+] LG TV found via HTTP: {ip_str}")
    
    def scan_network(self, network_range=None, max_threads=50):
        """Scan the network for LG TVs"""
        if network_range is None:
            network_range = self.get_local_network()
        
        print(f"Scanning network: {network_range}")
        print("Looking for LG TVs...")
        print("-" * 50)
        
        try:
            network = ipaddress.IPv4Network(network_range, strict=False)
            
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = []
                for ip in network.hosts():
                    future = executor.submit(self.scan_ip, ip)
                    futures.append(future)
                
                # Wait for all threads to complete
                for future in futures:
                    future.result()
                    
        except Exception as e:
            print(f"Error scanning network: {e}")
    
    def display_results(self):
        """Display scan results"""
        print("\n" + "="*50)
        print("SCAN RESULTS")
        print("="*50)
        
        if not self.found_tvs:
            print("No LG TVs found on the network.")
            print("\nTroubleshooting tips:")
            print("- Make sure TVs are powered on and connected to WiFi")
            print("- Check if TVs are on the same network segment")
            print("- Some TVs might have different port configurations")
        else:
            print(f"Found {len(self.found_tvs)} potential LG TV(s):")
            print()
            
            for i, tv in enumerate(self.found_tvs, 1):
                print(f"{i}. IP Address: {tv['ip']}")
                print(f"   Hostname: {tv['hostname']}")
                print(f"   MAC Address: {tv['mac_address']}")
                print(f"   WebOS Port (3001): {'Open' if tv['port_3001_open'] else 'Closed'}")
                print(f"   HTTP Detection: {'Success' if tv['http_detection'] else 'Failed'}")
                if tv['detected_url']:
                    print(f"   Detected URL: {tv['detected_url']}")
                print()

def main():
    scanner = LGTVScanner()
    
    # You can specify a custom network range if needed
    # scanner.scan_network("192.168.1.0/24")
    
    # Or let it auto-detect your network
    scanner.scan_network()
    
    scanner.display_results()

if __name__ == "__main__":
    # Check for required modules
    try:
        import netifaces
        import requests
    except ImportError as e:
        print(f"Missing required module: {e}")
        print("Please install required packages:")
        print("pip install netifaces requests")
        exit(1)
    
    main()