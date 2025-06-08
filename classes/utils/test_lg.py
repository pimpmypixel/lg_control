#!/usr/bin/env python3

import socket
import requests
import subprocess
import re
import json

def debug_lg_tv(ip):
    """Debug what's actually happening with the LG TV detection"""
    
    print(f"=== Debugging LG TV Detection for {ip} ===\n")
    
    # Test 1: Basic connectivity
    print("1. Testing basic connectivity...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((ip, 80))
        if result == 0:
            print("   ✅ Port 80 is open")
        else:
            print("   ❌ Port 80 is closed")
        sock.close()
    except Exception as e:
        print(f"   ❌ Error testing port 80: {e}")
    
    # Test 2: Comprehensive port scan
    print("\n2. Scanning common LG TV ports...")
    lg_ports = [80, 443, 1990, 3000, 3001, 8008, 8080, 8443, 9955, 36866]
    open_ports = []
    
    for port in lg_ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((ip, port))
            if result == 0:
                open_ports.append(port)
                print(f"   ✅ Port {port} is OPEN")
            sock.close()
        except Exception as e:
            pass
    
    if not open_ports:
        print("   ❌ No common LG TV ports found open")
    
    # Test 3: HTTP requests to open ports
    print("\n3. Testing HTTP responses on open ports...")
    for port in open_ports:
        for protocol in ['http', 'https']:
            try:
                url = f"{protocol}://{ip}:{port}"
                print(f"   Testing {url}...")
                response = requests.get(url, timeout=3, verify=False)
                print(f"   ✅ {url} responded with status {response.status_code}")
                
                # Check headers
                server = response.headers.get('Server', 'Unknown')
                print(f"      Server header: {server}")
                
                # Check content for LG indicators
                content = response.text[:500].lower()  # First 500 chars
                lg_keywords = ['webos', 'lg electronics', 'lg smart tv', 'netcast', 'lg corp']
                found_keywords = [kw for kw in lg_keywords if kw in content]
                if found_keywords:
                    print(f"      ✅ Found LG keywords: {found_keywords}")
                else:
                    print(f"      Content preview: {content[:100]}...")
                    
            except requests.exceptions.SSLError:
                print(f"   ⚠️  SSL Error on {protocol}://{ip}:{port}")
            except Exception as e:
                print(f"   ❌ Error accessing {protocol}://{ip}:{port}: {e}")
    
    # Test 4: MAC Address lookup
    print("\n4. Checking MAC address...")
    try:
        # Try different ARP commands
        commands = [['arp', '-n', ip], ['arp', '-a', ip]]
        mac_found = False
        
        for cmd in commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout:
                    print(f"   ARP result: {result.stdout.strip()}")
                    mac_match = re.search(r'([0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}', result.stdout)
                    if mac_match:
                        mac = mac_match.group(0).upper()
                        print(f"   ✅ MAC Address: {mac}")
                        
                        # Check OUI
                        oui = mac[:8].replace('-', ':')
                        lg_ouis = ['00:E0:91', '00:09:DF', 'A0:02:DC', 'B4:E6:2D', 
                                  '00:1C:62', '78:5D:C8', '00:1E:75', 'AC:5A:14',
                                  'A8:23:FE', '04:B1:67', '00:BB:35']
                        
                        if any(oui.startswith(lg_oui) for lg_oui in lg_ouis):
                            print(f"   ✅ OUI {oui} matches LG Electronics!")
                        else:
                            print(f"   ⚠️  OUI {oui} not in known LG list")
                        mac_found = True
                        break
            except Exception as e:
                continue
        
        if not mac_found:
            print("   ❌ Could not retrieve MAC address")
            print("   💡 Try: ping the IP first, then run again")
            
    except Exception as e:
        print(f"   ❌ Error getting MAC: {e}")
    
    # Test 5: Hostname lookup
    print("\n5. Checking hostname...")
    try:
        hostname = socket.gethostbyaddr(ip)[0]
        print(f"   ✅ Hostname: {hostname}")
        if any(keyword in hostname.lower() for keyword in ['lg', 'webos', 'smart', 'tv']):
            print("   ✅ Hostname suggests LG device!")
    except Exception as e:
        print(f"   ❌ Could not resolve hostname: {e}")
    
    # Test 6: WebOS API test
    print("\n6. Testing WebOS API endpoints...")
    webos_endpoints = [
        '/roap/api/auth',
        '/roap/api/command',
        '/udap/api/auth',
        '/udap/api/command'
    ]
    
    for port in [3000, 3001, 8080]:
        if port in open_ports:
            for endpoint in webos_endpoints:
                try:
                    url = f"http://{ip}:{port}{endpoint}"
                    response = requests.get(url, timeout=2)
                    print(f"   ✅ {url} responded: {response.status_code}")
                except:
                    pass
    
    print("\n" + "="*50)
    print("💡 SUGGESTIONS:")
    print("1. Make sure the TV is powered ON")
    print("2. Check if TV has network features enabled")
    print("3. Some LG TVs require pairing/authentication first")
    print("4. Try accessing the TV's web interface directly in a browser")
    if open_ports:
        print(f"5. Try these URLs in your browser:")
        for port in open_ports:
            print(f"   http://{ip}:{port}")

def main():
    ip = input("Enter LG TV IP address: ").strip()
    
    # Basic IP validation
    try:
        socket.inet_aton(ip)
    except socket.error:
        print("Invalid IP address format")
        return
    
    debug_lg_tv(ip)

if __name__ == "__main__":
    main()