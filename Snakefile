rule flow_metric_plots:
    input:
        # OmniBenchmark links the metrics input into "in/{dataset}.flow_metrics.json.gz"
        metrics="in/{dataset}.flow_metrics.json.gz"
    output:
        # OmniBenchmark supplies this exact full output path automatically.
        # DO NOT CHANGE — it gets overridden during execution.
        done="{output}"
    params:
        # ROOT = directory containing all metric files, 6 levels above the metric file
        root=lambda wildcards, input: str(Path(input.metrics).parents[6]),
        
        # OUTDIR = the folder where plots will be written
        # Same folder where the .done file will be placed
        outdir=lambda wildcards, output: str(Path(output.done).parent),

        # Python interpreter inside the conda env
        envpython="$(which python)"
    conda:
        "plot_env.yml"
    shell:
        r"""
        echo "[Plotting] Using root: {params.root}"
        echo "[Plotting] Writing PNG files to: {params.outdir}"

        mkdir -p {params.outdir}

        {params.envpython} visualize_flow_metrics.py \
            --root {params.root} \
            --outdir {params.outdir}

        echo "plots generated" > {output.done}
        """
