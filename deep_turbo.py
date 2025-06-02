import torch
import torch.nn as nn


# Y. Jiang, S. Kannan, H. Kim, S. Oh, H. Asnani, and P. Viswanath, “DEEPTURBO: Deep Turbo Decoder,”
# in 2019 IEEE 20th International Workshop on Signal Processing Advances in Wireless Communications (SPAWC), 2019, pp. 1–5.
class DeepTurboSISO(nn.Module):
    def __init__(self, input_size=2, embed_size=5, hidden_size=100, num_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        K = embed_size
        num_directions = 2 if bidirectional else 1

        # Bi-GRU for processing input sequence
        self.bigru = nn.GRU(
            input_size=input_size + K,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout
        )

        # Linear layer to produce posterior of shape (L, K)
        self.final_fc = nn.Linear(num_directions * hidden_size, K)

    def forward(self, y_sys, y_par, prior):
        # y_sys: systematic bits (batch, L, 1)
        # y_par: parity bits (batch, L, 1)
        # prior: prior information (batch, L, K)
        batch_size, L, _ = y_sys.size()

        # Concatenate inputs: systematic, parity, and prior
        inputs = torch.cat([y_sys, y_par, prior], dim=-1)  # (batch, L, 2+K)

        # Process through Bi-GRU
        gru_out, _ = self.bigru(inputs)  # (batch, L, 2*hidden_size)

        # Compute posterior
        posterior = self.final_fc(gru_out)  # (batch, L, K)

        # Extrinsic information: posterior - prior (ResNet-like shortcut)
        extrinsic = posterior - prior  # (batch, L, K)

        return posterior, extrinsic


class DeepTurboDecoder(nn.Module):
    def __init__(self, input_size=2, embed_size=5, hidden_size=100, num_layers=2, bidirectional=True, iter_times=6, dropout=0.3):
        super().__init__()
        self.K = embed_size
        self.num_iterations = iter_times

        # Non-shared SISO modules for each iteration (two SISOs per iteration)
        self.siso1_modules = nn.ModuleList([
            DeepTurboSISO(input_size=input_size,
                          embed_size=embed_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          dropout=dropout)
            for _ in range(iter_times)
        ])

        self.siso2_modules = nn.ModuleList([
            DeepTurboSISO(input_size=input_size,
                          embed_size=embed_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          dropout=dropout)
            for _ in range(iter_times)
        ])

        # Final output layer to produce LLRs
        self.final_fc = nn.Linear(self.K, 1)

    def forward(self, llr_sz, inter_pattern):
        batch_size, L, _ = llr_sz.size()
        inter_pattern_1 = inter_pattern.unsqueeze(-1)
        inter_pattern_k = inter_pattern.unsqueeze(-1).expand(batch_size, L, self.K)

        # y1: systematic bits (batch, L, 1)
        # y2: first parity bits (batch, L, 1)
        # y3: second parity bits (batch, L, 1)
        y1 = llr_sz[:, :, 0].unsqueeze(2)  # Systematic bits
        y2 = llr_sz[:, :, 1].unsqueeze(2)  # Parity bits from first encoder
        y3 = llr_sz[:, :, 2].unsqueeze(2)  # Parity bits from second encoder

        # Initialize prior and final posterior
        prior = torch.zeros(batch_size, L, self.K, device=llr_sz.device)
        final_posterior = torch.zeros(batch_size, L, self.K, device=llr_sz.device)

        # Main loop
        for it in range(self.num_iterations):
            # First SISO: takes y1, y2 and prior
            siso1 = self.siso1_modules[it]
            _, extrinsic1 = siso1(y1, y2, prior)

            # Interleave y1 and extrinsic information for second SISO
            inter_y1 = torch.gather(y1, dim=1, index=inter_pattern_1)
            inter_prior = torch.gather(extrinsic1, dim=1, index=inter_pattern_k)  # (batch, L, K)

            # Second SISO: takes interleaved y1, y3, and interleaved prior
            siso2 = self.siso2_modules[it]
            posterior2, extrinsic2 = siso2(inter_y1, y3, inter_prior)

            # Update prior for next iteration and deinterleaved prior
            prior.scatter_(dim=1, index=inter_pattern_k, src=extrinsic2)  # (batch, L, K)

            # Deinterleaved posterior2 for final posterior
            if it == self.num_iterations - 1:
                final_posterior.scatter_(dim=1, index=inter_pattern_k, src=posterior2)

        # Final posterior to LLR
        llr_c = self.final_fc(final_posterior).squeeze(-1)  # (batch, L)

        return llr_c
