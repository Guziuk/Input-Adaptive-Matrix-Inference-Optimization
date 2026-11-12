#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <nlohmann/json.hpp>

class Tokenizer {
private:
    std::unordered_map<int, std::string> vocab_by_id;
    std::unordered_map<std::string, int> vocab_by_string;
    std::unordered_map<std::string, int> merges;
    std::string space_token;
    
    struct Node {
        int token_id;
        Node* prev;
        Node* next;
        int orig_pos;
        double merge_prio;
        bool deleted;
        
        Node(int tid, Node* p, Node* n, int pos) 
            : token_id(tid), prev(p), next(n), orig_pos(pos), 
              merge_prio(0), deleted(false) {}
    };
    
    struct NodeComparator {
        bool operator()(Node* a, Node* b) const {
            return a->merge_prio > b->merge_prio; // Min heap
        }
    };
    
    std::string utf8_byte_to_hex(uint8_t c) {
        char buf[8];
        snprintf(buf, sizeof(buf), "<0x%02X>", c);
        return std::string(buf);
    }
    
    std::vector<int> map_characters_to_tokens(const std::string& text, 
                                          bool add_bos, 
                                          bool add_space) {
    std::vector<int> token_ids;
    
    if (add_bos) {
        token_ids.push_back(1);
    }
    
    std::string prompt = text;
    if (add_space) {
        prompt = " " + prompt;
    }
    
    // Replace spaces with space token (▁)
    std::string prompt_altered;
    for (char c : prompt) {
        if (c == ' ') {
            prompt_altered += space_token;
        } else {
            prompt_altered += c;
        }
    }
    
    // Go CHARACTER BY CHARACTER (like Swift), not greedy
    size_t i = 0;
    while (i < prompt_altered.size()) {
        // UTF-8 character handling
        int char_len = 1;
        unsigned char c = prompt_altered[i];
        
        // Determine UTF-8 character length
        if ((c & 0x80) == 0) {
            char_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4;
        }
        
        std::string character = prompt_altered.substr(i, char_len);
        
        // Try to find this character in vocab
        auto it = vocab_by_string.find(character);
        if (it != vocab_by_string.end()) {
            token_ids.push_back(it->second);
        } else {
            // Fallback to byte-level tokens
            for (int j = 0; j < char_len; j++) {
                uint8_t byte = static_cast<uint8_t>(prompt_altered[i + j]);
                std::string hex = utf8_byte_to_hex(byte);
                auto byte_it = vocab_by_string.find(hex);
                if (byte_it != vocab_by_string.end()) {
                    token_ids.push_back(byte_it->second);
                } else {
                    token_ids.push_back(0); // UNK
                }
            }
        }
        
        i += char_len;
    }
    
    return token_ids;
}
    
    std::string get_merge_identifier(int first_id, int second_id) {
        return vocab_by_id[first_id] + " " + vocab_by_id[second_id];
    }
    
    std::vector<int> perform_bpe_merges(std::vector<int> token_ids, size_t prompt_len) {
        if (token_ids.size() <= 1) return token_ids;
        
        std::priority_queue<Node*, std::vector<Node*>, NodeComparator> merge_queue;
        std::vector<Node*> all_nodes; // Track for cleanup
        
        auto add_to_queue = [&](Node* left_node) {
            if (left_node->next == nullptr) return;
            
            std::string merge_id = get_merge_identifier(left_node->token_id, 
                                                        left_node->next->token_id);
            auto it = merges.find(merge_id);
            if (it != merges.end()) {
                left_node->merge_prio = double(it->second) + 
                                       double(left_node->orig_pos) / double(prompt_len);
                merge_queue.push(left_node);
            }
        };
        
        // Create linked list
        Node* first_node = new Node(token_ids[0], nullptr, nullptr, 0);
        all_nodes.push_back(first_node);
        Node* prev_node = first_node;
        
        for (size_t i = 1; i < token_ids.size(); i++) {
            Node* curr_node = new Node(token_ids[i], prev_node, nullptr, i);
            all_nodes.push_back(curr_node);
            prev_node->next = curr_node;
            add_to_queue(prev_node);
            prev_node = curr_node;
        }
        
        // Perform merges
        while (!merge_queue.empty()) {
            Node* left = merge_queue.top();
            merge_queue.pop();
            
            if (left->deleted || left->next == nullptr || left->next->deleted) {
                continue;
            }
            
            left->deleted = true;
            left->next->deleted = true;
            
            // Handle prev node
            if (left->prev != nullptr) {
                Node* old_prev = left->prev;
                old_prev->deleted = true;
                Node* new_prev = new Node(old_prev->token_id, old_prev->prev, 
                                         old_prev->next, old_prev->orig_pos);
                all_nodes.push_back(new_prev);
                left->prev = new_prev;
                
                if (new_prev->prev != nullptr) {
                    new_prev->prev->next = new_prev;
                } else {
                    first_node = new_prev;
                }
            }
            
            // Create merged node
            std::string merged_token = vocab_by_id[left->token_id] + 
                                      vocab_by_id[left->next->token_id];
            
            auto it = vocab_by_string.find(merged_token);
            if (it == vocab_by_string.end()) {
                continue; // Merge not in vocab, skip
            }
            int merged_id = it->second;
            
            Node* result = new Node(merged_id, left->prev, left->next->next, 
                                   left->orig_pos);
            all_nodes.push_back(result);
            
            if (result->prev != nullptr) {
                result->prev->next = result;
                add_to_queue(result->prev);
            } else {
                first_node = result;
            }
            
            if (result->next != nullptr) {
                result->next->prev = result;
                add_to_queue(result);
            }
        }
        
        // Extract final tokens
        std::vector<int> merged_ids;
        Node* curr = first_node;
        while (curr != nullptr) {
            merged_ids.push_back(curr->token_id);
            curr = curr->next;
        }
        
        // Cleanup
        for (Node* node : all_nodes) {
            delete node;
        }
        
        return merged_ids;
    }
    
public:
    Tokenizer(const std::string& tokenizer_path = "tokenizer.json") {
        std::ifstream file(tokenizer_path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open tokenizer at: " + tokenizer_path);
        }
        
        nlohmann::json j;
        file >> j;
        
        // Load vocab
        for (auto& [token, id] : j["model"]["vocab"].items()) {
            int token_id = id.get<int>();
            vocab_by_string[token] = token_id;
            vocab_by_id[token_id] = token;
        }
        
        // Load merges
        int i = 0;
        for (auto& merge : j["model"]["merges"]) {
            merges[merge.get<std::string>()] = i + 1;
            i++;
        }
        
        space_token = "▁";
for (auto& [id, token] : vocab_by_id) {
    if (token == "▁") {
        space_token = token;
        std::cout << "Found space token at ID " << id << "\n";
        break;
    }
} // ▁ (Mistral-7B-v0.3)

        // Find the actual space token

        
        std::cout << "✓ Tokenizer loaded (" << vocab_by_id.size() 
                  << " tokens, " << merges.size() << " merges)\n";
    }
    
    std::vector<int> encode(const std::string& text, 
                           bool add_bos = true, 
                           bool add_space = true) {
        auto token_ids = map_characters_to_tokens(text, add_bos, add_space);
        return perform_bpe_merges(token_ids, text.length());
    }
    
    std::string decode(int token_id) {
        auto it = vocab_by_id.find(token_id);
        if (it != vocab_by_id.end()) {
            return it->second;
        }
        return "<unk>";
    }
    
    std::string decode(const std::vector<int>& tokens) {
    std::string result;
    for (int tok : tokens) {
        std::string s = decode(tok);
        
        // Skip special BOS/EOS tokens
        if (s == "<s>" || s == "</s>") {
            continue;
        }
        
        // Replace ▁ with space (▁ is 3 bytes in UTF-8: 0xE2 0x96 0x81)
        size_t pos = 0;
        while ((pos = s.find("▁", pos)) != std::string::npos) {
            s.replace(pos, 3, " ");
            pos += 1;
        }
        
        // Handle byte tokens like <0x0A>
        if (s.length() >= 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[s.length()-1] == '>') {
            std::string hex = s.substr(3, s.length() - 4);
            try {
                int byte = std::stoi(hex, nullptr, 16);
                result += static_cast<char>(byte);
            } catch (...) {
                result += s;
            }
        } else {
            result += s;
        }
    }
    return result;
}
};

#endif