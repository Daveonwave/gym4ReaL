# Gym4ReaL: A Suite for Benchmarking Real-World Reinforcement Learning

**Gym4ReaL** is a comprehensive suite of realistic environments designed to support the development and evaluation of RL algorithms that can operate in real-world scenarios. <br>
The suite includes a diverse set of tasks exposing RL algorithms to a variety of practical challenges.

Please refer to our [**Website**](https://daveonwave.github.io/gym4ReaL/) for further information about environments and **reproducibility**.

---

<h2>Coverage of <em>Characteristics</em> and <em>RL paradigms</em></h2>

<table style="border-collapse: collapse; width: 100%; text-align: center; font-family: sans-serif;">
  <thead>
    <tr style="background-color: #f0f0f0;">
      <th rowspan="2">Environment</th>
      <th colspan="6" style="background-color: #e0f7fa;">Characteristics</th>
      <th colspan="6" style="background-color: #fce4ec; border-left: 3px solid #555;">RL Paradigms</th>
    </tr>
    <tr>
      <th style="background-color: #e0f7fa;">Cont. States</th>
      <th style="background-color: #e0f7fa;">Cont. Actions</th>
      <th style="background-color: #e0f7fa;">Part. Observable</th>
      <th style="background-color: #e0f7fa;">Part. Controllable</th>
      <th style="background-color: #e0f7fa;">Non-Stationary</th>
      <th style="background-color: #e0f7fa;">Visual Input</th>
      <th style="border-left: 3px solid #555;background-color: #fce4ec;">Frequency Adaptation</th>
      <th style="background-color: #fce4ec;">Hierarchical RL</th>
      <th style="background-color: #fce4ec;">Risk-Averse</th>
      <th style="background-color: #fce4ec;">Imitation Learning</th>
      <th style="background-color: #fce4ec;">Provably Efficient</th>
      <th style="background-color: #fce4ec;">Multi-Objective RL</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #ffffff;">
      <td style=" font-weight: bold;"><em>DamEnv</em></td>
      <td>✅</td><td>✅</td><td></td><td>✅</td><td></td><td></td>
      <td style="border-left: 3px solid #555;"></td><td></td><td></td><td>✅</td><td></td><td>✅</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style=" font-weight: bold;"><em>ElevatorEnv</em></td>
      <td></td><td></td><td></td><td>✅</td><td></td><td></td>
      <td style="border-left: 3px solid #555;"></td><td></td><td></td><td></td><td>✅</td><td></td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style=" font-weight: bold;"><em>MicrogridEnv</em></td>
      <td>✅</td><td>✅</td><td></td><td>✅</td><td></td><td></td>
      <td style="border-left: 3px solid #555;">✅</td><td></td><td></td><td></td><td></td><td>✅</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="font-weight: bold;"><em>RoboFeederEnv</em></td>
      <td>✅</td><td>✅</td><td></td><td></td><td></td><td>✅</td>
      <td style="border-left: 3px solid #555;"></td><td>✅</td><td></td><td></td><td></td><td></td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style="font-weight: bold;"><em>TradingEnv</em></td>
      <td>✅</td><td></td><td>✅</td><td>✅</td><td>✅</td><td></td>
      <td style="border-left: 3px solid #555;">✅</td><td></td><td>✅</td><td></td><td></td><td></td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style=" font-weight: bold;"><em>WDSEnv</em></td>
      <td>✅</td><td></td><td></td><td>✅</td><td></td><td></td>
      <td style="border-left: 3px solid #555;"></td><td></td><td></td><td>✅</td><td></td><td>✅</td>
    </tr>
  </tbody>
</table>

---

## Folder structure

```
docs/                   # website and documentation
examples/               # example code for running each environment
gym4real/               # main Python package
    algorithms/
        {env}/          # per-env algorithms
    data/
        {env}/          # per-env data files
    envs/
        {env}/          # per-env modules
```

## Contributing

If you would like to add a new environment, propose bug fixes, or otherwise contribute to Gym4ReaL, please see the [Contributing Guide](CONTRIBUITING.md).

## License

Gym4ReaL is released under a [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). See the [LICENSE](LICENSE) file for the full terms.

## Citation

Please cite Gym4ReaL as

>

<details markdown="block">
<summary>BibTeX</summary>

```tex
@misc{salaorni2025gym4realsuitebenchmarkingrealworld,
      title={Gym4ReaL: A Suite for Benchmarking Real-World Reinforcement Learning}, 
      author={Davide Salaorni and Vincenzo De Paola and Samuele Delpero and Giovanni Dispoto and Paolo Bonetti and Alessio Russo and Giuseppe Calcagno and Francesco Trovò and Matteo Papini and Alberto Maria Metelli and Marco Mussi and Marcello Restelli},
      year={2025},
      eprint={2507.00257},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.00257}, 
}
```

</details>
