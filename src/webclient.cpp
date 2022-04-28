#include "webclient.h"

WebSocketClient::WebSocketClient(std::string host, std::string port, std::string url) {
    // Look up the domain name
    tcp::resolver resolver{m_ioc};
    auto const results = resolver.resolve(host, port);

    // Make the connection on the IP address we get from a lookup
    auto ep = net::connect(m_ws.next_layer(), results);

    // Update the host_ string. This will provide the value of the
    // Host HTTP header during the WebSocket handshake.
    // See https://tools.ietf.org/html/rfc7230#section-5.4
    host += ':' + std::to_string(ep.port());

    // Perform the websocket handshake
    m_ws.handshake(host, url);
}

std::string WebSocketClient::send(std::string s) {
    // Clear read buffer
    m_buffer.clear();

    // Send the message
    m_ws.write(net::buffer(s));

    // Read a message into our buffer
    m_ws.read(m_buffer);
    return beast::buffers_to_string(m_buffer.data());
}

WebSocketClient::~WebSocketClient() {
    // Close the WebSocket connection
    m_ws.close(websocket::close_code::normal);
}

HttpClient::HttpClient(std::string host, std::string port, std::string url) {
    // Look up the domain name
    tcp::resolver resolver{m_ioc};
    m_results = resolver.resolve(host, port);

    // Set up an HTTP POST request message
    m_req.method(beast::http::verb::post);
    m_req.target(url);
    m_req.set(http::field::host, host);
    m_req.set(http::field::content_type, "application/json");
}

std::string HttpClient::send(std::string s) {
    // Make the connection on the IP address we get from a lookup
    m_stream.connect(m_results);

    // Clear read buffer
    // m_buffer.clear();

    // Clear response
    m_res = {};

    // Prepare response
    m_req.body() = s;
    m_req.prepare_payload();

    // Send the HTTP request to the remote host
    http::write(m_stream, m_req);

    // Receive the HTTP response
    http::read(m_stream, m_buffer, m_res);

    // Gracefully close the socket
    beast::error_code ec;
    m_stream.socket().shutdown(tcp::socket::shutdown_both, ec);

    // not_connected happens sometimes, so don't bother reporting it.
    if (ec && ec != beast::errc::not_connected)
        throw beast::system_error{ec};

    // If we get here then the connection is closed gracefully
    return beast::buffers_to_string(m_res.body().data());
}

HttpClient::~HttpClient() {}
