#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 8080
#define BUFFER_SIZE 1024

int main() {
    int server_fd, client_fd;
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    char buffer[BUFFER_SIZE];
    char filename[64];
    int client_count = 0;

    // 1. Create socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // 2. Bind socket to IP and Port
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

    // 3. Listen for incoming connections
    listen(server_fd, 5);
    printf("Server listening on port %d...\n", PORT);

    // 4. Loop to accept multiple clients and save each client's info to a separate file
    while (1) {
        client_fd = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);
        if (client_fd < 0) {
            perror("Accept failed");
            continue;
        }

        client_count++;
        memset(buffer, 0, BUFFER_SIZE);
        
        // Receive system info string sent by client
        int bytes_read = read(client_fd, buffer, BUFFER_SIZE - 1);
        if (bytes_read > 0) {
            // Create a unique file for each client (e.g., client_1_info.txt)
            sprintf(filename, "client_%d_info.txt", client_count);
            FILE *fp = fopen(filename, "w");
            if (fp != NULL) {
                fprintf(fp, "Client IP: %s\n", inet_ntoa(address.sin_addr));
                fprintf(fp, "System Info:\n%s\n", buffer);
                fclose(fp);
                printf("Saved info to %s\n", filename);
            }
        }
        close(client_fd);
    }

    close(server_fd);
    return 0;
}