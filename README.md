# Gym4ReaL: A Suite for Benchmarking Real-World Reinforcement Learning

In recent years, _reinforcement learning_ (RL) has made remarkable progress, achieving superhuman performance in a wide range of simulated environments. As research moves toward deploying RL in real-world applications, the field faces a new set of challenges inherent to real-world settings, such as large state-action spaces, non-stationarity, and partial observability. Despite their importance, these challenges are often underexplored in current benchmarks, which tend to focus on idealized, fully observable, and stationary environments, often neglecting to incorporate real-world complexities explicitly. In this paper, we introduce _Gym4ReaL_, a comprehensive suite of realistic environments designed to support the development and evaluation of RL algorithms that can operate in real-world scenarios. The suite includes a diverse set of tasks that expose algorithms to a variety of practical challenges. Our experimental results show that, in these settings, standard RL algorithms confirm their competitiveness against rule-based benchmarks, motivating the development of new methods to fully exploit the potential of RL to tackle the complexities of real-world tasks.

<h3>Coverage of <em>Characteristics</em> and <em>RL paradigms</em></h3>

<table style="border-collapse: collapse; width: 100%; text-align: center; font-family: sans-serif;">
  <thead>
    <tr style="background-color: #f0f0f0;">
      <th rowspan="2">Environment</th>
      <th colspan="6" style="background-color: #e0f7fa;">Characteristics</th>
      <th colspan="6" style="background-color: #fce4ec; border-left: 3px solid #555;">RL Paradigms</th>
    </tr>
    <tr>
      <th style="background-color: #e0f7fa;">Continuous States</th>
      <th style="background-color: #e0f7fa;">Continuous Actions</th>
      <th style="background-color: #e0f7fa;">Partially Observable</th>
      <th style="background-color: #e0f7fa;">Partially Controllable</th>
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

[**Paper**]() | [**Website**](https://daveonwave.github.io/gym4ReaL/)

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

If you would like to add a new environment, propose bug fixes, or otherwise contribute to Gym4ReaL, please see the [Contributing Guide](CONTRIBUTING.md).

## License

Gym4ReaL is released under a [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). See the [LICENSE](LICENSE) file for the full terms.

## Citation

Please cite Gym4ReaL as

>

<details markdown="block">
<summary>BibTeX</summary>

```tex
@inproceedings{,
    title = {},
    author = {},
    year = ,
    month = ,
    booktitle = {T},
    address = {},
    url = {}
}
```

</details>
