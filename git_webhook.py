from http.server import BaseHTTPRequestHandler, HTTPServer
from subprocess import call


class PushHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            call(["git", "fetch", "origin"])
            call(["git", "reset", "--hard", "origin/dev"])
            self.send_response(202)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(bytes("ok", "utf-8"))
        except:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(bytes("Could not rebase", "utf-8"))


def run():
    server_address = ('', 6077)
    httpd = HTTPServer(server_address, PushHandler)
    print('Starting httpd...')
    httpd.serve_forever()

if __name__ == "__main__":
    run()
