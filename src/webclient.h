#ifndef WEBCLIENT_H
#define WEBCLIENT_H

#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/websocket.hpp>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

class WebSocketClient {
  public:
    WebSocketClient(std::string host, std::string port, std::string url);
    ~WebSocketClient();
    std::string send(std::string s);

  private:
    net::io_context m_ioc;                      // The io_context is required for all I/O
    websocket::stream<tcp::socket> m_ws{m_ioc}; // perform our I/O

    beast::flat_buffer m_buffer;
};

class HttpClient {
  public:
    HttpClient(std::string host, std::string port, std::string url);
    ~HttpClient();
    std::string send(std::string s);

  private:
    net::io_context m_ioc;             // The io_context is required for all I/O
    beast::tcp_stream m_stream{m_ioc}; // perform our I/O

    beast::flat_buffer m_buffer;
    http::request<http::string_body> m_req;
    http::response<http::dynamic_body> m_res;
    tcp::resolver::results_type m_results;
};

#endif // WEBCLIENT_H
