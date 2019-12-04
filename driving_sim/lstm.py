DEL

class LSTMNet(nn.Module):
        def __init__(self):
                    super(LSTMNet, self).__init__()
                            
                                    # Conv
                                            self.conv1 = nn.Conv2d(4, 64, 3, 1, 1)
                                                    self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
                                                            self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
                                                                    self.fc1 = nn.Linear(80*60*256, 100)
                                                                            self.fc2_1 = nn.Linear(100, 1)
                                                                                    self.fc2_2 = nn.Linear(100, 1)
                                                                                            self.drop = nn.Dropout(0.1)

                                                                                                    # Feed into GRU.
                                                                                                            self.lstm = nn.LSTM(100, 100, batch_first=True)
                                                                                                                    self.decoder = nn.Linear(100,2)
                                                                                                                            
                                                                                                                                    self.decoder.bias.data.zero_()
                                                                                                                                            
                                                                                                                                                    self.best_accuracy = -1
                                                                                                                                                        
                                                                                                                                                            def forward(self, x, hidden_state=None, cell_state=None):
                                                                                                                                                                        batch_size, sequence_length, C, H, W = x.size()
                                                                                                                                                                                c_in = x.view(batch_size * sequence_length, C, H, W)        
                                                                                                                                                                                        # Conv
                                                                                                                                                                                                x = self.pool(F.relu(self.conv1(c_in)))
                                                                                                                                                                                                        x = self.pool(F.relu(self.conv2(x)))
                                                                                                                                                                                                                x = x.view(-1, 16 * 5 * 5)
                                                                                                                                                                                                                        
                                                                                                                                                                                                                                # Feed into LSTM.
                                                                                                                                                                                                                                        r_in = x.view(batch_size, sequence_length, -1)        
                                                                                                                                                                                                                                                x, (hidden_state, cell_state) = self.lstm(x, hidden_state, cell_state)
                                                                                                                                                                                                                                                        x = self.drop(x)
                                                                                                                                                                                                                                                                x = self.decoder(x)

                                                                                                                                                                                                                                                                        return x, hidden_state, cell_state

                                                                                                                                                                                                                                                                        # Predefined loss function
                                                                                                                                                                                                                                                                            def loss(self, prediction, label, reduction='mean'):
                                                                                                                                                                                                                                                                                        loss_val = F.cross_entropy(prediction.view(-1, self.vocab_size), label.view(-1), reduction=reduction)
                                                                                                                                                                                                                                                                                                return loss_val

                                                                                                                                                                                                                                                                                                # Saves the current model
                                                                                                                                                                                                                                                                                                    def save_model(self, file_path, num_to_keep=1):
                                                                                                                                                                                                                                                                                                                pt_util.save(self, file_path, num_to_keep)

                                                                                                                                                                                                                                                                                                                    # Saves the best model so far
                                                                                                                                                                                                                                                                                                                        def save_best_model(self, accuracy, file_path, num_to_keep=1):
                                                                                                                                                                                                                                                                                                                                    if accuracy > self.best_accuracy:
                                                                                                                                                                                                                                                                                                                                                    self.save_model(file_path, num_to_keep)
                                                                                                                                                                                                                                                                                                                                                                self.best_accuracy = accuracy

                                                                                                                                                                                                                                                                                                                                                                    def load_model(self, file_path):
                                                                                                                                                                                                                                                                                                                                                                                pt_util.restore(self, file_path)

                                                                                                                                                                                                                                                                                                                                                                                    def load_last_model(self, dir_path):
                                                                                                                                                                                                                                                                                                                                                                                                return pt_util.restore_latest(self, dir_path)

