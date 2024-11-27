# Garak Architecture:

Runtime is a plugin based engine architecture:

### Where we are: what happens when garak runs?

`cli` merges run configuration in the following precedence:

* core configuration  
* plugin defaults  
* user site configuration file  
* command line arguments including additional configuration files

The primary `generator` and `evaluator` are instantiated and the set of `probes` and `detectors` are collected and passed to the `harness`.

`cli` starts the `run` initializing the reporting log data.

`cli` launches the configured `harness` detected based on configuration, if a `detector_spec` is provided all default detectors are overridden with the spec.

The `harness` instantiates each `probe` one at a time, passing in the `generator` under test, and executes the `probe`, which creates an `Attempt` for each prompt then sends the prompt to the generator and executes the `Attempt` by adding the response performing any pre and post processing on the response. The probe then logs the `Attempt` in the report and outputs the `Attempts` up to the harness. When configured for multiple generations the probe amplifies the response set buy either requesting multiple generations from the `generator` or by iterating the prompt for the requested number of unique calls to the `generator` under test.

The aggregate list of all attempts are then iterated and passed to the passed to each activated detector for the `probe`, emitting a hit log entry for any `Attempt` that meets the detection criteria.

The harness then marks each `Attempt` as complete and logs it again to the report.

The detectors return a result for each output and the harness again mutates each `Attempt` with detector specific results.

The resulting list of `Attempts` is passed to the evaluator which generates an analysis of the run places in the report log file.

`cli` then ends the run generating the html report output by processing the report log.

### Where we want to go

#### Parallelisation

While the current linear flow of the pipeline is somewhat rigid the requests made by the tool do have parallelization capabilities at the `probe` and `generator` levels.

Having said all this there are some significant components and design choices that are missed by this explanation. The plugins are designed with many flexible hooks to inject behaviors into the process and significant *meta* programming concepts allow for highly customizable configuration and deployment of `garak` to fit various hardware and resource constraints during runtime.

#### More powerful plugins

Each plugin type reduces the researcher or developer load significantly by only requiring the implementation of a limited set of core methods to create a new plugin. The structure of each plugin allows implementers flexibility to perform actions at each stage of the process without having to build and manage their own custom hooks for pre and post processing.

Configuration of plugins allows for injection of configuration concepts that can be encapsulated to ensure the overall pipeline does not need to impose strict requirement on all plugins, while still allowing an implementer to accept arbitrary configuration values as long at they can be provide primitive python `types` compatible with yaml and json safe load methods.

#### Better orchestration

The existing design offers locations that can be reworked to be more resilient and allow for more efficient scaling, due to clear separation of configuration and execution. All information needed to replicate a run is *known* at the start of the run although in the case of dynamic probes that emit prompts without predefined data the generation of those prompts is fully encapsulated in the plugin itself.

One such improvement might be to implement an `Attempt` queue and to shift the processing of `Attempts` to a more immutable `functional` pattern.