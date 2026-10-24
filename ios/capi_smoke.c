/* Smoke test of the C API on macOS:
 *   cargo build-capi --target aarch64-apple-darwin
 *   cc -O2 -Iinclude ios/capi_smoke.c target/aarch64-apple-darwin/release/liboxinfer.a -o /tmp/capi_smoke
 *   /tmp/capi_smoke MODEL [OPTIONS]      (e.g. Qwen/Qwen3.5-0.8B "quant=q8;mtp=1")
 * Sends two messages (the second is cancelled from its callback after 5 pieces), prints the
 * stats JSON after each, then checks error reporting. */
#include "oxinfer.h"
#include <stdio.h>
#include <string.h>

static int pieces;

static int on_text(void *user, const char *text, size_t len, int thinking) {
	(void)thinking;
	fwrite(text, 1, len, stdout);
	fflush(stdout);
	pieces++;
	int limit = *(int *)user;
	return limit > 0 && pieces >= limit; /* non-zero stops generation */
}

static int on_progress(void *user, const char *file, uint64_t done, uint64_t total, int index, int count) {
	(void)user;
	if (done == total)
		fprintf(stderr, "[%d/%d] %s: %llu bytes\n", index, count, file, (unsigned long long)done);
	return 0;
}

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr, "usage: %s MODEL [OPTIONS]\n", argv[0]);
		return 2;
	}
	printf("oxinfer %s\n", oxi_version());
	OxiEngine *e = oxi_engine_load(argv[1], argc > 2 ? argv[2] : "quant=q8", on_progress, NULL);
	if (!e) {
		fprintf(stderr, "load failed: %s\n", oxi_last_error());
		return 1;
	}
	char *info = oxi_engine_info_json(e);
	printf("info: %s\n", info);
	oxi_string_free(info);

	oxi_engine_set(e, "max_tokens", "64");
	int unlimited = 0, five = 5;
	int rc = oxi_engine_send(e, "Name three primary colors.", on_text, &unlimited);
	char *st = oxi_engine_stats_json(e);
	printf("\nfinish %d, stats: %s\n", rc, st);
	oxi_string_free(st);

	pieces = 0;
	rc = oxi_engine_send(e, "Now count from one to twenty in words.", on_text, &five);
	st = oxi_engine_stats_json(e);
	printf("\nfinish %d (2 = cancelled), stats: %s\n", rc, st);
	oxi_string_free(st);
	if (rc != 2)
		return 1;

	if (oxi_engine_set(e, "no_such_key", "1") >= 0)
		return 1;
	printf("expected error: %s\n", oxi_last_error());
	if (oxi_engine_load("/nonexistent/model", NULL, NULL, NULL) != NULL)
		return 1;
	printf("expected error: %s\n", oxi_last_error());
	oxi_engine_free(e);
	printf("ok\n");
	return 0;
}
