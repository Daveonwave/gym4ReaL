---
hide-toc: true
---

# Gym4ReaL

<p class="fs-6 fw-300">
Gymnasium-based benchmarking suite for testing RL algorithms on real-world scenarios
</p>

As research moves toward deploying _Reinforcement Learning_ in real-world applications, this field faces a new set of challenges, which are often underexplored in current benchmarks, which tend to focus on idealized, fully observable, and stationary environments.

We present **Gym4ReaL**, a comprehensive suite of realistic environments designed to support the development and evaluation of RL algorithms that can operate in real-world scenarios.

The suite includes a diverse set of tasks that expose algorithms to a variety of practical challenges to foster the development of RL to fully exploit its potential on real-world scenarios.

<p>
    <a href="https://github.com/Daveonwave/gym4ReaL" class="btn btn-primary">View it on GitHub</a>
</p>

---

<h2 style="font-weight: 500;">Coverage of <em>Characteristics</em> and <em>RL Paradigms</em></h2>

<div style="overflow-x: auto; max-width: 100%;">
<table style="border-collapse: collapse; width: 100%; text-align: center; font-size: 0.95rem; table-layout: fixed; border: 1px solid #ccc;">
  <colgroup>
    <col style="width: 130px;">
    <col span="12" style="width: 90px;">
  </colgroup>
  <thead>
    <tr style="background-color: var(--color-background-secondary, #f0f0f0);">
      <th rowspan="2" style="padding: 10px; border: 1px solid #ccc;">Env</th>
      <th colspan="6" style="background-color: #f0f0f0; color: var(--color-foreground-secondary, #f0f0f0); border: 1px solid #ccc;">Characteristics</th>
      <th colspan="6" style="background-color: #f0f0f0; color: var(--color-foreground-secondary, #f0f0f0); border: 1px solid #ccc; border-left: 3px solid #999;">RL Paradigms</th>
    </tr>
    <tr style="background-color: var(--color-background-secondary, #f0f0f0);">
      <th style="padding: 8px; border: 1px solid #ccc;">Cont. States</th>
      <th style="padding: 8px; border: 1px solid #ccc;">Cont. Actions</th>
      <th style="padding: 8px; border: 1px solid #ccc;">Part. Obs.</th>
      <th style="padding: 8px; border: 1px solid #ccc;">Part. Ctrl.</th>
      <th style="padding: 8px; border: 1px solid #ccc;">Non-Stat.</th>
      <th style="padding: 8px; border: 1px solid #ccc;">Visual In.</th>
      <th style="padding: 8px; border: 1px solid #ccc; border-left: 3px solid #999;">Freq. Adapt.</th>
      <th style="padding: 8px; border: 1px solid #ccc;">Hier. RL</th>
      <th style="padding: 8px; border: 1px solid #ccc;">Risk-Av.</th>
      <th style="padding: 8px; border: 1px solid #ccc;">Imitation</th>
      <th style="padding: 8px; border: 1px solid #ccc;">Prov. Eff.</th>
      <th style="padding: 8px; border: 1px solid #ccc;">Multi-Obj.</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #fff;">
      <td style="padding: 10px; border: 1px solid #ccc;"><em>Dam</em></td>
      <td>✅</td><td>✅</td><td></td><td>✅</td><td></td><td></td>
      <td style="border-left: 3px solid #999;"></td><td></td><td></td><td>✅</td><td></td><td>✅</td>
    </tr>
    <tr style="background-color: #f6f6f6;">
      <td style="padding: 10px; border: 1px solid #ccc;"><em>Elevator</em></td>
      <td></td><td></td><td></td><td>✅</td><td></td><td></td>
      <td style="border-left: 3px solid #999;"></td><td></td><td></td><td></td><td>✅</td><td></td>
    </tr>
    <tr style="background-color: #fff;">
      <td style="padding: 10px; border: 1px solid #ccc;"><em>Microgrid</em></td>
      <td>✅</td><td>✅</td><td></td><td>✅</td><td></td><td></td>
      <td style="border-left: 3px solid #999;">✅</td><td></td><td></td><td></td><td></td><td>✅</td>
    </tr>
    <tr style="background-color: #f6f6f6;">
      <td style="padding: 10px; border: 1px solid #ccc;"><em>RoboFeeder</em></td>
      <td>✅</td><td>✅</td><td></td><td></td><td></td><td>✅</td>
      <td style="border-left: 3px solid #999;"></td><td>✅</td><td></td><td></td><td></td><td></td>
    </tr>
    <tr style="background-color: #fff;">
      <td style="padding: 10px; border: 1px solid #ccc;"><em>Trading</em></td>
      <td>✅</td><td></td><td>✅</td><td>✅</td><td>✅</td><td></td>
      <td style="border-left: 3px solid #999;">✅</td><td></td><td>✅</td><td></td><td></td><td></td>
    </tr>
    <tr style="background-color: #f6f6f6;">
      <td style="padding: 10px; border: 1px solid #ccc;"><em>WDS</em></td>
      <td>✅</td><td></td><td></td><td>✅</td><td></td><td></td>
      <td style="border-left: 3px solid #999;"></td><td></td><td></td><td>✅</td><td></td><td>✅</td>
    </tr>
  </tbody>
</table>
</div>

Gym4ReaL is released under [Apache-2.0 license](http://www.apache.org/licenses/LICENSE-2.0).

```{toctree}
:hidden:
:maxdepth: 3

Home <self>
team
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Environments

dam
elevator
microgrid
robofeeder
trading
wds
```

```{toctree}
:hidden:
:caption: API Reference

api/gym4real/envs/index
api/gym4real/algortithms/index

genindex
modindex
```
